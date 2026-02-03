// Multi-dimensional bin packing via MIP (Mixed Integer Programming).
//
// Data model (Azure Packing Trace):
//   - "Items" are VMs, queried from the `vm` table at a given timestamp.
//   - "Bin types" are physical machine types, identified by `machineId`.
//   - Each VM has a `vmTypeId` (flavor). The `vmType` table maps each
//     (vmTypeId, machineId) pair to a 5-dimensional resource vector
//     (core, memory, hdd, ssd, nic), all normalized to [0, 1].
//     Not every VM flavor can run on every machine type -- only
//     combinations present in `vmType` are valid placements.
//
// Key optimization: VMs sharing a vmTypeId are interchangeable (identical
// resource footprint on every machine type). Instead of one binary variable
// per individual VM, we use one integer variable per unique VM type,
// representing how many VMs of that type are placed on each machine instance.
// At t=0, 860k active VMs collapse to ~245 unique types.
//
// MIP formulation (aggregated):
//   Sets:
//     G       = unique VM type groups (by vmTypeId), each with count_g VMs
//     T       = machine types (machineId values)
//     T(g)    = machine types valid for group g
//     J(t)    = instance indices {0..max_j(t)} per machine type t
//
//   Decision variables:
//     n[g,t,j] integer in [0, count_g] -- # of VMs from group g on instance (t,j)
//     y[t,j]   in {0,1}               -- instance j of type t is used
//
//   Objective:
//     minimize  sum_{t,j} y[t,j]
//
//   Constraints:
//     (1) Assignment:       sum_{t in T(g), j in J(t)} n[g,t,j] = count_g  for all g
//     (2) Capacity (per d): sum_g resource_d[g,t] * n[g,t,j] <= y[t,j]     for all t,j,d
//     (3) Symmetry break:   y[t,j] <= y[t,j-1]                              for j >= 1
//
//   Capacity constraint (2) also implies linking: if any n[g,t,j] > 0 and
//   resource_d > 0, then y[t,j] >= resource_d * n > 0, forcing y[t,j] = 1.
//
// Upper bound on max_j(t):
//   For each machine type t, we sum the resource requirements of all active
//   VMs assignable to t (per dimension) and take ceil(max across dims).
//   This is conservative: it assumes every assignable VM could land on type t.
//
// Previous formulation (per-VM binary variables):
//   Used binary x[v,t,j] in {0,1} for each individual VM, machine type,
//   and instance. This led to millions of variables even for moderate VM
//   counts and could not scale beyond ~230 VMs.
//
//   | VMs | Unique types | MIP vars  | Solve time | Machines |
//   |-----|--------------|-----------|------------|----------|
//   | 50  | -            | 3,980     | 0.8s       | 11       |
//   | 100 | -            | 8,582     | 2.2s       | 13       |
//   | 200 | -            | 21,218    | 10.1s      | 21       |
//   | 220 | -            | 23,902    | 46.6s      | 22       |
//   | 230 | -            | 26,228    | 11.8s      | 23       |
//   | 500 | -            | 129,380   | killed     | -        |
//   (above times are from the old per-VM binary formulation)
//
// Current formulation (aggregated integer variables):
//   Replaces per-VM binaries with per-vmTypeId integers n[g,t,j].
//   This is an exact reformulation (same optimal objective).
//
//   | VMs   | Unique types | MIP vars | Solve time | Machines |
//   |-------|--------------|----------|------------|----------|
//   | 100   | 7            | 1,156    | 0.4s       | 13       |
//   | 500   | 8            | 2,992    | 18.3s      | 58       |
//   | 1,000 | 8            | 4,918    | 29.1s      | 106      |

use std::collections::HashMap;
use std::error::Error;

use clap::Parser;
use good_lp::{
    variable, Expression, ProblemVariables, Solution, SolverModel,
};
use good_lp::solvers::highs::highs;
use indicatif::{ProgressBar, ProgressStyle};
use rusqlite::Connection;

#[derive(Parser)]
#[command(name = "binpack", about = "Solve bin packing MIP for Azure VM traces")]
struct Args {
    /// Path to the SQLite database file
    #[arg(short, long, default_value = "packing_trace_zone_a_v1.sqlite")]
    db: String,

    /// Timestamp at which to snapshot active VMs
    #[arg(short, long)]
    timestamp: f64,

    /// Time limit for the MIP solver in seconds (0 = no limit)
    #[arg(long, default_value_t = 300.0)]
    time_limit: f64,

    /// MIP relative gap tolerance (e.g. 0.01 for 1%)
    #[arg(long, default_value_t = 0.0)]
    mip_gap: f32,

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

struct VmTypeGroup {
    vm_type_id: i64,
    count: usize,
}

struct NEntry {
    machine_id: i64,
    instance_j: usize,
    var: good_lp::Variable,
    placement: VmTypePlacement,
}

fn progress_bar(len: u64, message: &str) -> ProgressBar {
    let bar = ProgressBar::new(len);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {wide_bar} {pos}/{len} [{elapsed_precise}]")
            .unwrap(),
    );
    bar.set_message(message.to_string());
    bar
}

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
    // If max_vms is set, we need to load individual VMs first to truncate,
    // then aggregate. Otherwise we can aggregate directly in SQL.
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
        println!("  {} active VMs (truncated to {}), {} unique types", total, vm_types.len(), groups.len());
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

    // -- Preprocessing --

    // Group placements by vm_type_id
    let mut placements_by_vm_type: HashMap<i64, Vec<VmTypePlacement>> = HashMap::new();
    for p in &all_placements {
        placements_by_vm_type
            .entry(p.vm_type_id)
            .or_default()
            .push(p.clone());
    }

    // Count VMs assignable to each machine type, collect reachable machine types
    let mut vms_per_machine_type: HashMap<i64, usize> = HashMap::new();
    for g in &vm_type_groups {
        if let Some(placements) = placements_by_vm_type.get(&g.vm_type_id) {
            for p in placements {
                *vms_per_machine_type.entry(p.machine_id).or_default() += g.count;
            }
        }
    }

    let mut machine_type_ids: Vec<i64> = vms_per_machine_type.keys().copied().collect();
    machine_type_ids.sort();

    // Compute max instances per machine type (upper bound).
    // Sum resource requirements of all active VMs assignable to each machine type,
    // per dimension. ceil(max across dimensions) is a lower bound on machines needed.
    let mut max_instances: HashMap<i64, usize> = HashMap::new();
    for &t in &machine_type_ids {
        let mut sum_core = 0.0_f64;
        let mut sum_mem = 0.0_f64;
        let mut sum_hdd = 0.0_f64;
        let mut sum_ssd = 0.0_f64;
        let mut sum_nic = 0.0_f64;

        for g in &vm_type_groups {
            if let Some(placements) = placements_by_vm_type.get(&g.vm_type_id) {
                for p in placements {
                    if p.machine_id == t {
                        let c = g.count as f64;
                        sum_core += p.core * c;
                        sum_mem += p.memory * c;
                        sum_hdd += p.hdd * c;
                        sum_ssd += p.ssd * c;
                        sum_nic += p.nic * c;
                    }
                }
            }
        }

        let max_sum = sum_core.max(sum_mem).max(sum_hdd).max(sum_ssd).max(sum_nic);
        let bound = max_sum.ceil().max(1.0) as usize;
        max_instances.insert(t, bound);
    }

    println!("Machine types in use: {}", machine_type_ids.len());
    for &t in &machine_type_ids {
        println!(
            "  type {}: {} assignable VMs, max {} instances",
            t, vms_per_machine_type[&t], max_instances[&t]
        );
    }

    // -- MIP construction --

    let mut vars = ProblemVariables::new();

    // y[t][j] variables
    let mut y: HashMap<i64, Vec<good_lp::Variable>> = HashMap::new();
    for &t in &machine_type_ids {
        let max_j = max_instances[&t];
        let y_vec: Vec<good_lp::Variable> = (0..max_j)
            .map(|_| vars.add(variable().binary()))
            .collect();
        y.insert(t, y_vec);
    }

    // n[g][...] variables per VM type group
    let bar = progress_bar(vm_type_groups.len() as u64, "Creating n variables");
    let mut n_by_group: Vec<Vec<NEntry>> = Vec::with_capacity(vm_type_groups.len());
    for g in &vm_type_groups {
        bar.inc(1);
        let mut entries = Vec::new();
        if let Some(placements) = placements_by_vm_type.get(&g.vm_type_id) {
            for placement in placements {
                let t = placement.machine_id;
                let max_j = max_instances[&t];
                for j in 0..max_j {
                    let var = vars.add(variable().integer().min(0).max(g.count as f64));
                    entries.push(NEntry {
                        machine_id: t,
                        instance_j: j,
                        var,
                        placement: placement.clone(),
                    });
                }
            }
        }
        n_by_group.push(entries);
    }
    bar.finish_and_clear();

    let total_n: usize = n_by_group.iter().map(|e| e.len()).sum();
    let total_y: usize = y.values().map(|v| v.len()).sum();
    println!("MIP variables: {} n + {} y = {} total", total_n, total_y, total_n + total_y);

    // Objective: minimize sum of all y
    let mut objective = Expression::with_capacity(total_y);
    for y_vec in y.values() {
        for &y_tj in y_vec {
            objective += y_tj;
        }
    }

    let mut model = vars.minimise(objective).using(highs);
    if args.verbose {
        model.set_verbose(true);
    }
    if args.time_limit > 0.0 {
        model = model.set_time_limit(args.time_limit);
    }
    if args.mip_gap > 0.0 {
        model = model.set_mip_rel_gap(args.mip_gap)?;
    }
    if let Some(threads) = args.threads {
        model = model.set_threads(threads);
    }

    // Constraint 1: each VM type group fully assigned
    let bar = progress_bar(vm_type_groups.len() as u64, "Assignment constraints");
    for (g_idx, g) in vm_type_groups.iter().enumerate() {
        bar.inc(1);
        let sum: Expression = n_by_group[g_idx].iter().map(|e| e.var).sum();
        model.add_constraint(sum.eq(g.count as i32));
    }
    bar.finish_and_clear();

    // Pre-index: (machine_id, j) -> list of (group_idx, entry_idx)
    let mut entries_by_instance: HashMap<(i64, usize), Vec<(usize, usize)>> = HashMap::new();
    for (g_idx, entries) in n_by_group.iter().enumerate() {
        for (e_idx, entry) in entries.iter().enumerate() {
            entries_by_instance
                .entry((entry.machine_id, entry.instance_j))
                .or_default()
                .push((g_idx, e_idx));
        }
    }

    // Constraint 2: capacity per dimension per (t, j)
    let total_instances: usize = max_instances.values().sum();
    let bar = progress_bar((total_instances * 5) as u64, "Capacity constraints");
    for &t in &machine_type_ids {
        let max_j = max_instances[&t];
        let y_vec = &y[&t];

        for j in 0..max_j {
            let mut core_expr = Expression::default();
            let mut mem_expr = Expression::default();
            let mut hdd_expr = Expression::default();
            let mut ssd_expr = Expression::default();
            let mut nic_expr = Expression::default();

            if let Some(instance_entries) = entries_by_instance.get(&(t, j)) {
                for &(g_idx, e_idx) in instance_entries {
                    let entry = &n_by_group[g_idx][e_idx];
                    core_expr.add_mul(entry.placement.core, entry.var);
                    mem_expr.add_mul(entry.placement.memory, entry.var);
                    hdd_expr.add_mul(entry.placement.hdd, entry.var);
                    ssd_expr.add_mul(entry.placement.ssd, entry.var);
                    nic_expr.add_mul(entry.placement.nic, entry.var);
                }
            }

            let y_tj = y_vec[j];
            model.add_constraint(core_expr.leq(y_tj));
            model.add_constraint(mem_expr.leq(y_tj));
            model.add_constraint(hdd_expr.leq(y_tj));
            model.add_constraint(ssd_expr.leq(y_tj));
            model.add_constraint(nic_expr.leq(y_tj));
            bar.inc(5);
        }
    }
    bar.finish_and_clear();

    // Constraint 3: symmetry breaking y[t][j] <= y[t][j-1]
    for &t in &machine_type_ids {
        let y_vec = &y[&t];
        for j in 1..y_vec.len() {
            model.add_constraint(Expression::from(y_vec[j]).leq(y_vec[j - 1]));
        }
    }

    // -- Solve --
    println!("Solving ({} VMs, {} types, {} vars)...", total_vms, vm_type_groups.len(), total_n + total_y);
    let solve_start = std::time::Instant::now();
    let solution = model.solve()?;
    let solve_time = solve_start.elapsed();

    // -- Extract results --
    let mut total_machines = 0usize;
    let mut machines_by_type: Vec<(i64, usize)> = Vec::new();

    for &t in &machine_type_ids {
        let y_vec = &y[&t];
        let count = y_vec.iter().filter(|&&v| solution.value(v) > 0.5).count();
        if count > 0 {
            machines_by_type.push((t, count));
            total_machines += count;
        }
    }

    println!("\n=== Result ===");
    println!("Total machines: {}", total_machines);
    println!("Status: {:?}", solution.status());
    println!("Solve time: {:.3}s", solve_time.as_secs_f64());
    println!("\nBy machine type:");
    for (machine_id, count) in &machines_by_type {
        println!("  machine type {}: {} instances", machine_id, count);
    }

    Ok(())
}
