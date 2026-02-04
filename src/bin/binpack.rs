// Optimal multi-dimensional vector bin packing for Azure VM packing traces.
//
// Problem:
//   Given a snapshot of active VMs at a point in time, find the minimum
//   number of physical machines (bins) needed to host all VMs.  Each VM
//   has a type (vmTypeId) that determines its 5-dimensional resource
//   footprint (core, memory, hdd, ssd, nic — all normalized to [0,1]).
//   Each physical machine has a type (machineId) with unit capacity in
//   every dimension.  Not every VM type fits on every machine type; only
//   (vmTypeId, machineId) pairs present in the vmType table are valid.
//   VMs sharing a vmTypeId are interchangeable.
//
//   At t=0 the Azure trace has ~860k active VMs collapsing to ~245
//   unique types across 35 machine types.  A direct MIP formulation
//   (one variable per VM × machine-type × instance) cannot scale past
//   ~1000 VMs.  The goal is to find the provably optimal packing for
//   the full dataset.
//
//   For this problem, only Cargo.{toml, lock}, src/bin/binpack.rs and
//   the dataset file are needed. Ignore the other files in this repo.



// Approach:
// Algorithm — column generation (Dantzig-Wolfe decomposition):
//
//   Master LP:  minimize total machines.
//     Variables: λ_p ≥ 0 for each packing pattern p.
//     Constraints: for each VM type group g, the patterns must
//                  collectively provide exactly count_g slots.
//
//   Pricing subproblem (per machine type): multi-dimensional bounded
//   knapsack that finds the most valuable new packing pattern given
//   dual prices from the master LP.  ~200 integer vars, 5 constraints.
//
//   After the LP relaxation converges, a restricted MIP over the
//   generated columns yields an integer-optimal solution.  For vector
//   bin packing the LP gap is almost always 0 or 1, so
//   ceil(LP) == MIP proves optimality.

use std::collections::HashMap;
use std::error::Error;

use clap::Parser;
use highs::{ColProblem, Sense, HighsModelStatus};
use rusqlite::Connection;

#[derive(Parser)]
#[command(name = "binpack", about = "Solve VM bin packing via column generation")]
struct Args {
    /// Path to the SQLite database file
    #[arg(short, long, default_value = "packing_trace_zone_a_v1.sqlite")]
    db: String,

    /// Timestamp at which to snapshot active VMs
    #[arg(short, long)]
    timestamp: f64,

    /// Time limit for the restricted MIP solver in seconds
    #[arg(long, default_value_t = 300.0)]
    time_limit: f64,

    /// MIP relative gap tolerance (e.g. 0.01 for 1%)
    #[arg(long, default_value_t = 0.0)]
    mip_gap: f64,

    /// Enable verbose solver output
    #[arg(short, long)]
    verbose: bool,

    /// Only consider the first N active VMs (for testing)
    #[arg(long)]
    max_vms: Option<usize>,

    /// Number of solver threads (default: all CPUs)
    #[arg(long)]
    threads: Option<u32>,
}

#[derive(Debug, Clone)]
struct VmTypePlacement {
    vm_type_id: i64,
    machine_id: i64,
    core: f64,
    memory: f64,
    hdd: f64,
    ssd: f64,
    nic: f64,
}

impl VmTypePlacement {
    fn resources(&self) -> [f64; 5] {
        [self.core, self.memory, self.hdd, self.ssd, self.nic]
    }
}

struct VmTypeGroup {
    vm_type_id: i64,
    count: usize,
}

/// A packing pattern: what goes on one physical machine.
#[derive(Debug, Clone)]
struct Pattern {
    machine_id: i64,
    /// Sparse (group_index, count) pairs.
    items: Vec<(usize, f64)>,
}

/// Compatibility entry: one VM type group on one machine type.
struct CompatEntry {
    group_idx: usize,
    resources: [f64; 5],
    max_per_machine: usize,
}

// ---------------------------------------------------------------------------
// Data loading (reused from original)
// ---------------------------------------------------------------------------

fn load_vm_type_placements(conn: &Connection) -> Result<Vec<VmTypePlacement>, Box<dyn Error>> {
    let mut stmt = conn.prepare(
        "SELECT vmTypeId, machineId, core, memory, COALESCE(hdd, 0.0), ssd, nic FROM vmType",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(VmTypePlacement {
            vm_type_id: row.get(0)?,
            machine_id: row.get(1)?,
            core: row.get(2)?,
            memory: row.get(3)?,
            hdd: row.get(4)?,
            ssd: row.get(5)?,
            nic: row.get(6)?,
        })
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

fn load_active_vm_type_counts(
    conn: &Connection,
    timestamp: f64,
    max_vms: Option<usize>,
) -> Result<(Vec<VmTypeGroup>, usize), Box<dyn Error>> {
    if let Some(max) = max_vms {
        let mut stmt = conn.prepare(
            "SELECT vmTypeId FROM vm WHERE starttime <= ?1 AND (endtime IS NULL OR endtime > ?1)",
        )?;
        let mut vm_types: Vec<i64> = stmt
            .query_map([timestamp], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;
        let total = vm_types.len();
        vm_types.truncate(max);
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for vt in &vm_types {
            *counts.entry(*vt).or_default() += 1;
        }
        let groups: Vec<VmTypeGroup> = counts
            .into_iter()
            .map(|(vm_type_id, count)| VmTypeGroup { vm_type_id, count })
            .collect();
        println!(
            "  {} active VMs (truncated to {}), {} unique types",
            total,
            vm_types.len(),
            groups.len()
        );
        Ok((groups, vm_types.len()))
    } else {
        let mut stmt = conn.prepare(
            "SELECT vmTypeId, COUNT(*) FROM vm \
             WHERE starttime <= ?1 AND (endtime IS NULL OR endtime > ?1) \
             GROUP BY vmTypeId",
        )?;
        let groups: Vec<VmTypeGroup> = stmt
            .query_map([timestamp], |row| {
                Ok(VmTypeGroup {
                    vm_type_id: row.get(0)?,
                    count: row.get::<_, i64>(1)? as usize,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let total: usize = groups.iter().map(|g| g.count).sum();
        println!("  {} active VMs, {} unique types", total, groups.len());
        Ok((groups, total))
    }
}

// ---------------------------------------------------------------------------
// Master LP
// ---------------------------------------------------------------------------

struct MasterResult {
    objective: f64,
    duals: Vec<f64>,
    lambdas: Vec<f64>,
    status: HighsModelStatus,
}

fn solve_master_lp(
    groups: &[VmTypeGroup],
    patterns: &[Pattern],
    verbose: bool,
) -> MasterResult {
    let mut pb = ColProblem::new();

    // Covering constraints: Σ_p a_{g,p} · λ_p = count_g  for each group g
    let rows: Vec<_> = groups
        .iter()
        .map(|g| {
            let b = g.count as f64;
            pb.add_row(b..=b)
        })
        .collect();

    // Pattern columns (continuous): obj coeff = 1, lower bound = 0
    for pattern in patterns {
        let coeffs: Vec<_> = pattern
            .items
            .iter()
            .map(|&(g_idx, count)| (rows[g_idx], count))
            .collect();
        pb.add_column(1.0, 0.0.., &coeffs);
    }

    let mut model = pb.optimise(Sense::Minimise);
    if !verbose {
        model.set_option("output_flag", false);
    }
    let solved = model.solve();
    let solution = solved.get_solution();

    MasterResult {
        objective: solved.objective_value(),
        duals: solution.dual_rows().to_vec(),
        lambdas: solution.columns().to_vec(),
        status: solved.status(),
    }
}

// ---------------------------------------------------------------------------
// Pricing subproblem (one per machine type)
// ---------------------------------------------------------------------------

/// Greedy pricing heuristic: pack items by value density (π / max resource).
/// Much faster than MIP (~microseconds vs ~40ms per call).
/// Any feasible packing with profit > 1 is a valid improving column.
/// Returns (Option<pattern>, greedy_profit).
/// The pattern is Some only if profit > 1 + ε (improving column).
/// The profit is always returned so we can decide whether MIP verification is worthwhile.
fn solve_pricing_greedy(
    machine_id: i64,
    compat: &[CompatEntry],
    duals: &[f64],
) -> (Option<Pattern>, f64) {
    if compat.is_empty() {
        return (None, 0.0);
    }

    // Sort by value density = π_g / max resource dimension (descending)
    let mut order: Vec<usize> = (0..compat.len())
        .filter(|&i| duals[compat[i].group_idx] > 1e-12)
        .collect();
    order.sort_by(|&a, &b| {
        let density_a = duals[compat[a].group_idx]
            / compat[a].resources.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
        let density_b = duals[compat[b].group_idx]
            / compat[b].resources.iter().copied().fold(0.0_f64, f64::max).max(1e-12);
        density_b.partial_cmp(&density_a).unwrap()
    });

    let mut remaining = [1.0_f64; 5];
    let mut items: Vec<(usize, f64)> = Vec::new();
    let mut profit = 0.0_f64;

    for &idx in &order {
        let entry = &compat[idx];
        let max_fit = (0..5)
            .map(|d| {
                if entry.resources[d] > 1e-12 {
                    (remaining[d] / entry.resources[d]).floor() as usize
                } else {
                    usize::MAX
                }
            })
            .min()
            .unwrap_or(0)
            .min(entry.max_per_machine);

        if max_fit > 0 {
            for d in 0..5 {
                remaining[d] -= entry.resources[d] * max_fit as f64;
            }
            profit += duals[entry.group_idx] * max_fit as f64;
            items.push((entry.group_idx, max_fit as f64));
        }
    }

    if profit > 1.0 + 1e-6 && !items.is_empty() {
        (Some(Pattern { machine_id, items }), profit)
    } else {
        (None, profit)
    }
}

/// Tight upper bound on pricing via LP relaxation (continuous variables).
/// If this is ≤ 1, no integer column can be improving → skip MIP.
fn pricing_lp_bound(compat: &[CompatEntry], duals: &[f64]) -> f64 {
    if compat.is_empty() {
        return 0.0;
    }
    let mut pb = ColProblem::new();
    let cap_rows: Vec<_> = (0..5).map(|_| pb.add_row(..=1.0)).collect();
    for entry in compat {
        let pi = duals[entry.group_idx];
        let coeffs: Vec<_> = (0..5)
            .map(|d| (cap_rows[d], entry.resources[d]))
            .collect();
        pb.add_column(pi, 0.0..=entry.max_per_machine as f64, &coeffs);
    }
    let mut model = pb.optimise(Sense::Maximise);
    model.set_option("output_flag", false);
    model.solve().objective_value()
}

/// Exact pricing via MIP (used only when LP relaxation says a column might exist).
fn solve_pricing_exact(
    machine_id: i64,
    compat: &[CompatEntry],
    duals: &[f64],
) -> Option<(Pattern, f64)> {
    if compat.is_empty() {
        return None;
    }

    // Use LP relaxation as a tight filter
    let ub = pricing_lp_bound(compat, duals);
    if ub <= 1.0 + 1e-6 {
        return None;
    }

    let mut pb = ColProblem::new();

    // 5 capacity constraints: Σ_g res_d[g] · a_g ≤ 1.0
    let cap_rows: Vec<_> = (0..5).map(|_| pb.add_row(..=1.0)).collect();

    let mut group_indices: Vec<usize> = Vec::with_capacity(compat.len());
    for entry in compat {
        let pi = duals[entry.group_idx];
        let coeffs: Vec<_> = (0..5)
            .map(|d| (cap_rows[d], entry.resources[d]))
            .collect();
        pb.add_integer_column(pi, 0.0..=entry.max_per_machine as f64, &coeffs);
        group_indices.push(entry.group_idx);
    }

    let mut model = pb.optimise(Sense::Maximise);
    model.set_option("output_flag", false);
    model.set_option("time_limit", 2.0);
    let solved = model.solve();
    let obj = solved.objective_value();

    if obj > 1.0 + 1e-6 {
        let solution = solved.get_solution();
        let col_vals = solution.columns();
        let items: Vec<(usize, f64)> = group_indices
            .iter()
            .enumerate()
            .filter_map(|(i, &g_idx)| {
                let val: f64 = col_vals[i];
                if val > 0.5 {
                    Some((g_idx, val.round()))
                } else {
                    None
                }
            })
            .collect();
        if !items.is_empty() {
            return Some((Pattern { machine_id, items }, obj - 1.0));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Restricted MIP (integer λ over generated columns)
// ---------------------------------------------------------------------------

fn solve_restricted_mip(
    groups: &[VmTypeGroup],
    patterns: &[Pattern],
    lp_lambdas: &[f64],
    time_limit: f64,
    mip_gap: f64,
    verbose: bool,
    threads: Option<u32>,
) -> MasterResult {
    let mut pb = ColProblem::new();

    // Covering constraints (≥): each group must be fully placed.
    // We use ≥ instead of = here because with integer λ, exact equality
    // may be infeasible (e.g. pattern packs 7 VMs but need exactly 100).
    let rows: Vec<_> = groups
        .iter()
        .map(|g| pb.add_row(g.count as f64..))
        .collect();

    // Integer λ columns with upper bounds derived from LP solution
    for (i, pattern) in patterns.iter().enumerate() {
        let coeffs: Vec<_> = pattern
            .items
            .iter()
            .map(|&(g_idx, count)| (rows[g_idx], count))
            .collect();
        // Upper bound: 2× the LP value (rounded up) + 1, to give solver room
        let lp_val = if i < lp_lambdas.len() { lp_lambdas[i] } else { 0.0 };
        let ub = (lp_val.ceil() as usize * 2 + 1).max(1);
        pb.add_integer_column(1.0, 0.0..=ub as f64, &coeffs);
    }

    let mut model = pb.optimise(Sense::Minimise);
    if !verbose {
        model.set_option("output_flag", false);
    }
    if time_limit > 0.0 {
        model.set_option("time_limit", time_limit);
    }
    if mip_gap > 0.0 {
        model.set_option("mip_rel_gap", mip_gap);
    }
    if let Some(t) = threads {
        model.set_option("threads", t as i32);
    }

    let solved = model.solve();
    let solution = solved.get_solution();

    MasterResult {
        objective: solved.objective_value(),
        duals: solution.dual_rows().to_vec(),
        lambdas: solution.columns().to_vec(),
        status: solved.status(),
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("Opening database: {}", args.db);
    let conn = Connection::open(&args.db)?;

    println!("Loading VM type placements...");
    let all_placements = load_vm_type_placements(&conn)?;
    println!("  {} placement entries", all_placements.len());

    println!("Loading active VMs at timestamp {}...", args.timestamp);
    let (vm_type_groups, total_vms) =
        load_active_vm_type_counts(&conn, args.timestamp, args.max_vms)?;

    if vm_type_groups.is_empty() {
        println!("No active VMs at this timestamp.");
        return Ok(());
    }

    // -- Preprocessing -------------------------------------------------------

    let n_groups = vm_type_groups.len();

    let group_index: HashMap<i64, usize> = vm_type_groups
        .iter()
        .enumerate()
        .map(|(i, g)| (g.vm_type_id, i))
        .collect();

    // Placements by vm_type_id
    let mut placements_by_type: HashMap<i64, Vec<VmTypePlacement>> = HashMap::new();
    for p in &all_placements {
        placements_by_type
            .entry(p.vm_type_id)
            .or_default()
            .push(p.clone());
    }

    // Reachable machine types
    let machine_type_ids: Vec<i64> = {
        let mut set = std::collections::HashSet::new();
        for g in &vm_type_groups {
            if let Some(ps) = placements_by_type.get(&g.vm_type_id) {
                for p in ps {
                    set.insert(p.machine_id);
                }
            }
        }
        let mut v: Vec<i64> = set.into_iter().collect();
        v.sort();
        v
    };

    // Compatibility map: machine_id → [CompatEntry]
    let compat_by_machine: HashMap<i64, Vec<CompatEntry>> = {
        let mut map: HashMap<i64, Vec<CompatEntry>> = HashMap::new();
        for &t in &machine_type_ids {
            let mut entries = Vec::new();
            for g in &vm_type_groups {
                if let Some(ps) = placements_by_type.get(&g.vm_type_id) {
                    for p in ps {
                        if p.machine_id == t {
                            let res = p.resources();
                            let max_per_dim = res
                                .iter()
                                .map(|&r| {
                                    if r > 1e-12 {
                                        (1.0 / r).floor() as usize
                                    } else {
                                        usize::MAX
                                    }
                                })
                                .min()
                                .unwrap_or(0);
                            let max_a = max_per_dim.min(g.count);
                            if max_a > 0 {
                                entries.push(CompatEntry {
                                    group_idx: group_index[&g.vm_type_id],
                                    resources: res,
                                    max_per_machine: max_a,
                                });
                            }
                        }
                    }
                }
            }
            if !entries.is_empty() {
                map.insert(t, entries);
            }
        }
        map
    };

    println!(
        "{} VM type groups, {} reachable machine types, {} total VMs",
        n_groups,
        machine_type_ids.len(),
        total_vms
    );

    // -- Initial columns -----------------------------------------------------
    // For each group, create a trivial "pure" pattern (pack only this type).
    let mut patterns: Vec<Pattern> = Vec::new();
    for (g_idx, g) in vm_type_groups.iter().enumerate() {
        if let Some(ps) = placements_by_type.get(&g.vm_type_id) {
            if let Some(p) = ps.first() {
                let res = p.resources();
                let max_fit = res
                    .iter()
                    .map(|&r| {
                        if r > 1e-12 {
                            (1.0 / r).floor() as usize
                        } else {
                            usize::MAX
                        }
                    })
                    .min()
                    .unwrap_or(1)
                    .max(1)
                    .min(g.count);
                patterns.push(Pattern {
                    machine_id: p.machine_id,
                    items: vec![(g_idx, max_fit as f64)],
                });
            }
        }
    }
    println!("Initial columns: {}", patterns.len());

    // -- Column generation ---------------------------------------------------
    let cg_start = std::time::Instant::now();
    let mut iteration = 0;
    let mut lp_bound = 0.0_f64;
    let mut lp_lambdas: Vec<f64> = Vec::new();

    loop {
        iteration += 1;

        let master = solve_master_lp(&vm_type_groups, &patterns, args.verbose);

        if master.status != HighsModelStatus::Optimal {
            eprintln!(
                "Master LP not optimal (status {:?}) at iteration {}",
                master.status, iteration
            );
            break;
        }

        lp_bound = master.objective;
        lp_lambdas = master.lambdas;

        // Phase 1: greedy pricing (fast — microseconds per machine type)
        let mut new_patterns: Vec<Pattern> = Vec::new();
        let mut best_rc = 0.0_f64;

        for &t in &machine_type_ids {
            if let Some(compat) = compat_by_machine.get(&t) {
                let (pattern, profit) =
                    solve_pricing_greedy(t, compat, &master.duals);
                if let Some(p) = pattern {
                    best_rc = best_rc.max(profit - 1.0);
                    new_patterns.push(p);
                }
            }
        }

        // Phase 2: if greedy found nothing, verify with exact MIP pricing.
        // The LP relaxation filter (inside solve_pricing_exact) eliminates
        // machine types where no integer column can possibly be improving.
        if new_patterns.is_empty() {
            for &t in &machine_type_ids {
                if let Some(compat) = compat_by_machine.get(&t) {
                    if let Some((pattern, rc)) =
                        solve_pricing_exact(t, compat, &master.duals)
                    {
                        best_rc = best_rc.max(rc);
                        new_patterns.push(pattern);
                    }
                }
            }
        }

        if iteration % 100 == 1 || new_patterns.is_empty() {
            println!(
                "  iter {:>4}: LP = {:>12.4}, cols = {:>5}, new = {:>2}, best_rc = {:.6}",
                iteration,
                master.objective,
                patterns.len(),
                new_patterns.len(),
                best_rc
            );
        }

        if new_patterns.is_empty() {
            break;
        }

        patterns.extend(new_patterns);
    }

    let cg_time = cg_start.elapsed();
    println!(
        "\nColumn generation converged: {} iterations, {:.3}s",
        iteration,
        cg_time.as_secs_f64()
    );
    println!(
        "LP lower bound: {:.4}  (ceil = {})",
        lp_bound,
        lp_bound.ceil() as usize
    );
    println!("Generated columns: {}", patterns.len());

    // -- Restricted MIP ------------------------------------------------------
    println!(
        "\nSolving restricted MIP ({} cols, {} groups)...",
        patterns.len(),
        n_groups
    );
    let mip_start = std::time::Instant::now();
    let mip = solve_restricted_mip(
        &vm_type_groups,
        &patterns,
        &lp_lambdas,
        args.time_limit,
        args.mip_gap,
        args.verbose,
        args.threads,
    );
    let mip_time = mip_start.elapsed();

    // -- Results -------------------------------------------------------------
    if mip.status != HighsModelStatus::Optimal
        && mip.status != HighsModelStatus::ObjectiveBound
        && mip.status != HighsModelStatus::ReachedTimeLimit
    {
        eprintln!("Restricted MIP status: {:?}", mip.status);
        eprintln!("This may require more columns or branch-and-price.");
        return Ok(());
    }
    let total_machines = mip.objective.round() as usize;
    let lp_ceil = lp_bound.ceil() as usize;

    println!("\n=== Result ===");
    println!("Total machines: {}", total_machines);
    println!("LP lower bound: {:.4}  (ceil = {})", lp_bound, lp_ceil);
    if total_machines == lp_ceil {
        println!("Optimality: PROVEN OPTIMAL");
    } else {
        println!(
            "Optimality: gap = {} machines (branch-and-price may be needed)",
            total_machines - lp_ceil
        );
    }
    println!("MIP status: {:?}", mip.status);
    println!(
        "Solve time: CG {:.3}s + MIP {:.3}s = {:.3}s total",
        cg_time.as_secs_f64(),
        mip_time.as_secs_f64(),
        (cg_time + mip_time).as_secs_f64()
    );

    // Per machine type breakdown
    let mut machines_by_type: HashMap<i64, usize> = HashMap::new();
    for (p_idx, &lambda) in mip.lambdas.iter().enumerate() {
        if lambda > 0.5 {
            *machines_by_type
                .entry(patterns[p_idx].machine_id)
                .or_default() += lambda.round() as usize;
        }
    }
    let mut type_counts: Vec<(i64, usize)> = machines_by_type.into_iter().collect();
    type_counts.sort_by_key(|&(t, _)| t);
    println!("\nBy machine type:");
    for (t, count) in &type_counts {
        println!("  type {:>2}: {} instances", t, count);
    }

    Ok(())
}
