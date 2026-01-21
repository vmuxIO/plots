use indicatif::{ProgressBar, ProgressStyle};
use rusqlite::{Connection, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct VmType {
    pub vm_type_id: i64,
    pub machine_id: i64,
    pub core: f64,
    pub memory: f64,
    pub hdd: f64,
    pub ssd: f64,
    pub nic: f64,
}

#[derive(Debug, Clone)]
pub struct VmRequest {
    pub vm_id: i64,
    pub vm_type_id: i64,
    pub starttime: f64,
    pub endtime: Option<f64>,
    pub time_sorter: Option<f64>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StartResult {
    Ok,
    CoreBottleneck,
    MemoryBottleneck,
    HddBottleneck,
    SsdBottleneck,
    NicBottleneck,
}

#[derive(Debug, Clone)]
pub struct VM {
    pub vm_id: i64,
    pub vm_type_id: i64,
}

// Physical machine instance
#[derive(Debug, Clone)]
pub struct Machine {
    pub machine_id: i64,
    pub vms: Vec<VM>,

    // utilization
    core: f64,
    memory: f64,
    hdd: f64,
    ssd: f64,
    nic: f64,
}

impl Machine {
    pub fn new(machine_id: i64) -> Self {
        Machine {
            machine_id,
            vms: Vec::new(),
            core: 0.0,
            memory: 0.0,
            hdd: 0.0,
            ssd: 0.0,
            nic: 0.0,
        }
    }

    pub fn start_vm2(&mut self, vm: VM, core: f64, memory: f64, hdd: f64, ssd: f64, nic: f64) -> StartResult {
        let _core = self.core + core;
        let _memory = self.memory + memory;
        let _hdd = self.hdd + hdd;
        let _ssd = self.ssd + ssd;
        let _nic = self.nic + nic;

        if _core > 1.0 {
            return StartResult::CoreBottleneck;
        }
        if _memory > 1.0 {
            return StartResult::MemoryBottleneck;
        }
        if _hdd > 1.0 {
            return StartResult::HddBottleneck;
        }
        if _ssd > 1.0 {
            return StartResult::SsdBottleneck;
        }
        if _nic > 1.0 {
            return StartResult::NicBottleneck;
        }
        self.vms.push(vm);
        self.core = _core;
        self.memory = _memory;
        self.hdd = _hdd;
        self.ssd = _ssd;
        self.nic = _nic;

        return StartResult::Ok;
    }

    pub fn stop_vm(&mut self, vm_id: i64, machine_type: &VmType) -> bool {
        let matches = self.vms.iter().filter(|vm| vm.vm_id == vm_id);
        if matches.count() == 0 {
            return false;
        }
        self.core -= machine_type.core;
        self.memory -= machine_type.memory;
        self.hdd -= machine_type.hdd;
        self.ssd -= machine_type.ssd;
        self.nic -= machine_type.nic;
        // TODO shouldnt we remove the VM from self.vms?
        return true;
    }
}

#[derive(Debug, Clone)]
pub struct FirstFitDecreasing {
    iteration: usize,

    machine_types: HashMap<i64, Vec<Machine>>, // machine_type -> instantiated physical machines
    #[allow(nonstandard_style)]
    vmId_to_machine: HashMap<i64, (i64, usize)>, // vm_id -> (machine_type, machine_index)

    cores: f64,
    memory: f64,
    hdd: f64,
    ssd: f64,
    nic: f64,
}

impl FirstFitDecreasing {
    pub fn new() -> Self {
        FirstFitDecreasing {
            iteration: 0,
            machine_types: HashMap::new(),
            vmId_to_machine: HashMap::new(),
            cores: 0.0,
            memory: 0.0,
            hdd: 0.0,
            ssd: 0.0,
            nic: 0.0,
        }
    }

    pub fn pool_size(&self) -> usize {
        self.machine_types.values().map(|machines| machines.len()).sum()
    }

    pub fn schedule(&mut self, request: &VmRequest, vm_types: &Vec<VmType>) {
        let vm_type = request.vm_type_id;
        let machine_type_candidates = vm_types
            .iter()
            .filter(|vt| vt.vm_type_id == vm_type)
            .collect::<Vec<&VmType>>();

        // TODO select optimal candidate
        let optimal_type = match false {
            true => {
                // TODO implement fragmentation
                unreachable!();
            },
            false => machine_type_candidates.first().expect("VM types are complete. We always have one."),
        };

        let core = optimal_type.core;
        let memory = optimal_type.memory;
        let hdd = optimal_type.hdd;
        let ssd = optimal_type.ssd;
        let nic = optimal_type.nic;

        // bookkeeping
        let machines = self.machine_types.entry(optimal_type.machine_id).or_insert(Vec::new());
        let vm = VM { vm_id: request.vm_id, vm_type_id: request.vm_type_id };
        let mut started = None;
        let mut machine_idx = None;
        for (idx, machine) in machines.iter_mut().enumerate() {
            started = Some(machine.start_vm2(vm.clone(), core, memory, hdd, ssd, nic));
            // TODO measure bottlenecks
            if let Some(StartResult::Ok) = started {
                machine_idx = Some(idx);
                break;
            }
        }

        // create new machine if necessary
        if started.is_none() || started.as_ref().unwrap() != &StartResult::Ok {
            machines.push(Machine::new(optimal_type.machine_id));
            started = Some(machines.last_mut().expect("that we just pushed something")
                .start_vm2(vm.clone(), core, memory, hdd, ssd, nic));
            machine_idx = Some(machines.len() - 1);
        }
        assert_eq!(started.unwrap(), StartResult::Ok);
        assert!(machine_idx.is_some());

        // collect bottleneck statistics
        // TODO

        self.cores += optimal_type.core;
        self.memory += optimal_type.memory;
        self.hdd += optimal_type.hdd;
        self.ssd += optimal_type.ssd;
        self.nic += optimal_type.nic;

        self.vmId_to_machine.insert(vm.vm_id, (optimal_type.machine_id, machine_idx.expect("that we asserted Some")));
    }

    pub fn unschedule(&mut self, vm_request: &VmRequest, vm_types: &Vec<VmType>) {
        let vm_type = vm_request.vm_type_id;
        let vm_id = vm_request.vm_id;
        let (machine_id, machine_idx) = match self.vmId_to_machine.get(&vm_id) {
            Some(data) => data.clone(),
            None => panic!("Trying to unschedule VM {} which was never scheduled?", vm_id),
        };
        let vm_usage = vm_types.iter().find(|vt| vt.vm_type_id == vm_type && vt.machine_id == machine_id);
        let vm_usage = vm_usage.unwrap();
        let machine = self.machine_types.get_mut(&machine_id).unwrap().get_mut(machine_idx).unwrap();
        let _ = machine.stop_vm(vm_id, vm_usage) || panic!("Failed to stop VM {} on machine {}", vm_id, machine_id);

        self.cores -= vm_usage.core;
        self.memory -= vm_usage.memory;
        self.hdd -= vm_usage.hdd;
        self.ssd -= vm_usage.ssd;
        self.nic -= vm_usage.nic;

        if machine.vms.len() == 0 {
            self.machine_types.get_mut(&machine_id).unwrap().remove(machine_idx);
        }
        self.vmId_to_machine.remove(&vm_id);
    }

    pub fn simulate(&mut self, vm_requests: &Vec<VmRequest>, vm_types: &Vec<VmType>) {
        let checkpoint_interval = 100_000;
        let mut time: f64 = f64::MIN; // log the newest point in time reached by our simulation
        let mut time_series_time = Vec::with_capacity(vm_requests.len());
        let mut time_series_pool_size = Vec::with_capacity(vm_requests.len());
        let mut time_series_cores = Vec::with_capacity(vm_requests.len());
        let mut time_series_memory = Vec::with_capacity(vm_requests.len());
        let mut time_series_hdd = Vec::with_capacity(vm_requests.len());
        let mut time_series_ssd = Vec::with_capacity(vm_requests.len());
        let mut time_series_nic = Vec::with_capacity(vm_requests.len());

        // let mut time_series_ = Vec::with_capacity(vm_requests.len()); TODO

        let bar = progress_bar(vm_requests.len() as u64);
        for (index, request) in vm_requests.iter().enumerate() {
            bar.inc(1);
            if index < self.iteration {
                continue; // skip requests that we already processed (e.g. loaded checkpoint)
            }

            if let Some(time_sorter) = request.time_sorter && request.starttime == time_sorter {
                self.schedule(request, vm_types);
            } else if request.endtime == request.time_sorter {
                self.unschedule(request, vm_types);
            } else {
                panic!("Invalid VmRequest: {:?}", request);
            }

            // TODO checkpointing
            if checkpoint_interval > 0 && index % checkpoint_interval == 0 {
                println!("Checkpoint: {} hosts", self.pool_size());
            }

            let time_sorter = match request.time_sorter {
                Some(t) => t,
                None => continue,
            };
            if time == f64::MIN {
                time = time_sorter;
                continue;
            }
            if time_sorter > time {
                // we have processed all events for this current point in time
                // (multiple rows can have the same time for which we aggregate results)
                time = time_sorter;
                time_series_time.push(time);
                time_series_pool_size.push(self.pool_size());
                // cluster usage (1-strading) statistics
                time_series_cores.push(self.cores);
                time_series_memory.push(self.memory);
                time_series_hdd.push(self.hdd);
                time_series_ssd.push(self.ssd);
                time_series_nic.push(self.nic);
                // TODO find bottlenecks
            }
        }
        bar.finish();
    }
}

pub fn load_data(sqlite_file: &str) -> Result<(Vec<VmRequest>, Vec<VmType>)> {
    let conn = Connection::open(sqlite_file)?;

    // Load VM types
    let mut stmt = conn.prepare(
        "SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType"
    )?;
    let vm_types: Vec<VmType> = stmt
        .query_map([], |row| {
            Ok(VmType {
                vm_type_id: row.get(0)?,
                machine_id: row.get(1)?,
                core: row.get(2)?,
                memory: row.get(3)?,
                hdd: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                ssd: row.get(5)?,
                nic: row.get(6)?,
            })
        })?
        .collect::<Result<Vec<_>>>()?;

    // Load VM requests
    let mut stmt = conn.prepare(
        "SELECT vmId, vmTypeId, starttime, endtime FROM vm"
    )?;
    let vms: Vec<(i64, i64, f64, Option<f64>)> = stmt
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get::<_, Option<f64>>(2)?.ok_or(rusqlite::Error::InvalidQuery)?, row.get(3)?))
        })?
        .collect::<Result<Vec<_>>>()?;

    let mut vm_requests: Vec<VmRequest> = Vec::with_capacity(vms.len() * 2);

    for (vm_id, vm_type_id, starttime, endtime) in &vms {
        // Add start event
        vm_requests.push(VmRequest {
            vm_id: *vm_id,
            vm_type_id: *vm_type_id,
            starttime: *starttime,
            endtime: *endtime,
            time_sorter: Some(*starttime),
        });
        // Add end event
        vm_requests.push(VmRequest {
            vm_id: *vm_id,
            vm_type_id: *vm_type_id,
            starttime: *starttime,
            endtime: *endtime,
            time_sorter: *endtime,
        });
    }

    // Sort by time_sorter (Nones at the end)
    vm_requests.sort_by(|a, b| {
        match (a.time_sorter, b.time_sorter) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    Ok((vm_requests, vm_types))
}

fn progress_bar(len: u64) -> ProgressBar {
    let bar = ProgressBar::new(len);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {percent}% {pos}/{len} [{elapsed_precise}] ({per_sec})")
            .unwrap(),
    );
    bar
}

fn main() {
    println!("Loading data");
    let (vm_requests, vm_types) = load_data("packing_trace_zone_a_v1.sqlite").unwrap();
    println!("Loaded {} VM requests and {} VM types", vm_requests.len(), vm_types.len());

    // TODO rank VMs

    let mut scheduler = FirstFitDecreasing::new();
    scheduler.simulate(&vm_requests, &vm_types);

}
