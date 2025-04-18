import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass

def log(msg):
    print(msg, flush=True)

# Connect to the database
conn = sqlite3.connect('packing_trace_zone_a_v1.sqlite')

# Load VM requests

log("Loading data")
vm_types = pd.read_sql_query("SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType", conn)

a = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, starttime as time_sorter FROM vm", conn)
b = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, endtime as time_sorter FROM vm", conn)
vm_requests = pd.concat([a, b]).sort_values("time_sorter") # one line/event for each start and end of a VM
vm_requests = vm_requests.head(2_000_000)


@dataclass
class VM():
    vmId: int
    vmTypeId: int

@dataclass
class Machine():
    machineId: int # machine type. Not unique!
    vms: List[Any]

    # utilization
    core: float = 0
    memory: float = 0
    hdd: float = 0
    ssd: float = 0
    nic: float = 0

    def start_vm(self, vm: VM, machine_type: pd.Series) -> bool:
        _core = self.core + machine_type["core"]
        _memory = self.memory + machine_type["memory"]
        _hdd = self.hdd + machine_type["hdd"]
        _ssd = self.ssd + machine_type["ssd"]
        _nic = self.nic + machine_type["nic"]
        if _core > 1 or _memory > 1 or _hdd > 1 or _ssd > 1 or _nic > 1:
            return False
        self.vms.append(vm)
        self.core = _core
        self.memory = _memory
        self.hdd = _hdd
        self.ssd = _ssd
        self.nic = _nic
        return True

    def stop_vm(self, vmId: int, machine_type: pd.Series) -> bool:
        matches = [vm for vm in self.vms if vm.vmId == vmId]
        if len(matches) == 0:
            return False
        self.core -= machine_type["core"]
        self.memory -= machine_type["memory"]
        self.hdd -= machine_type["hdd"]
        self.ssd -= machine_type["ssd"]
        self.nic -= machine_type["nic"]
        return True

class Scheduler():
    machine_types: Dict[int, List[Machine]] = dict()
    vmId_to_machine: Dict[int, Tuple[int, int]] = dict() # vmId -> (machine_type, machine_idx)

    cores: float = 0
    memory: float = 0
    hdd: float = 0
    ssd: float = 0
    nic: float = 0

    fragmented: bool

    def __init__(self, fragmented: bool = False):
        self.fragmented = fragmented
        self.machine_types = dict()
        self.vmId_to_machine = dict()


    def schedule(self, vm_request: pd.Series):
        vm_type = vm_request["vmTypeId"]
        machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm_type]

        # TODO select optimal candidate
        if self.fragmented:
            pool_fragment = int(vm_request["vmId"]) % 3
            optimal_type = machine_type_candidates.iloc[pool_fragment % len(machine_type_candidates)]
        else:
            optimal_type = machine_type_candidates.iloc[0]

        # bookkeeping
        self.machine_types[optimal_type["machineId"]] = self.machine_types.get(optimal_type["machineId"], [])
        machines = self.machine_types[optimal_type["machineId"]]
        vm = VM(vmId=vm_request["vmId"], vmTypeId=vm_request["vmTypeId"])
        started = False
        machine_idx = None
        for i in range(len(machines) - 1, 0, -1):
            machine = machines[i]
            started = machine.start_vm(vm, optimal_type)
            if started:
                machine_idx = i
                break
        if not started:
            machines.append(Machine(machineId=optimal_type["machineId"], vms=[]))
            started = machines[-1].start_vm(vm, optimal_type)
            machine_idx = len(machines) - 1
        assert started

        self.cores += optimal_type["core"]
        self.memory += optimal_type["memory"]
        self.hdd += optimal_type["hdd"]
        self.ssd += optimal_type["ssd"]
        self.nic += optimal_type["nic"]

        self.vmId_to_machine[vm.vmId] = (optimal_type["machineId"], machine_idx)
        # machines[vm_request["vmTypeId"]]
        pass

    def unschedule(self, vm_request: pd.Series):
        vm_type = vm_request["vmTypeId"]
        vmId = vm_request["vmId"]
        machineId, machine_idx = self.vmId_to_machine[vmId]
        vm_usage = vm_types[(vm_types["vmTypeId"] == vm_type) & (vm_types["machineId"] == machineId)]
        assert len(vm_usage) == 1
        vm_usage = vm_usage.iloc[0]
        machine = self.machine_types[machineId][machine_idx]
        assert machine.stop_vm(vmId, vm_usage)

        self.cores -= vm_usage["core"]
        self.memory -= vm_usage["memory"]
        self.hdd -= vm_usage["hdd"]
        self.ssd -= vm_usage["ssd"]
        self.nic -= vm_usage["nic"]

        if len(machine.vms) == 0:
            del self.machine_types[machineId][machine_idx]
        del self.vmId_to_machine[vmId]

        pass


    def pool_size(self) -> int:
        return sum(len(machines) for machines in self.machine_types.values())


    def dump(self):
        print("Machines:")
        for machineId, machines in self.machine_types.items():
            for machine in machines:
                print(f" - Machine type {machineId}:")
                print(f"   {len(machine.vms)} VMs, core={machine.core}, memory={machine.memory}, hdd={machine.hdd}, ssd={machine.ssd}, nic={machine.nic}")

    # def stranded_resources(self) -> Tuple[int, int, int, int, int]:
    #     core = 0
    #     memory = 0
    #     hdd = 0
    #     ssd = 0
    #     nic = 0
    #     for machines in self.machine_types.values():
    #         for machine in machines:
    #             core += machine.core
    #             memory += machine.memory
    #             hdd += machine.hdd
    #             ssd += machine.ssd
    #             nic += machine.nic
    #     return core, memory, hdd, ssd, nic

    def simulate(self):
        time = None
        time_series_time = []
        time_series_pool_size = []
        stranded_cores = [] # wasted (unused) core resources

        for index, row in tqdm(vm_requests.iterrows(), total=len(vm_requests)):
            if row["starttime"] == row["time_sorter"]:
                scheduler.schedule(row)
            elif row["endtime"] == row["time_sorter"]:
                scheduler.unschedule(row)
            else:
                assert False

            if row["time_sorter"] is None:
                continue
            if time is None:
                time = row["time_sorter"]
                continue
            if row["time_sorter"] > time:
                time = row["time_sorter"]
                time_series_time += [time]
                time_series_pool_size += [scheduler.pool_size()]

        df = pd.DataFrame({
            "time": time_series_time,
            "pool_size": time_series_pool_size,
            "cores": self.cores,
            "memory": self.memory,
            "hdd": self.hdd,
            "ssd": self.ssd,
            "nic": self.nic,
        })
        return df

filename = "plot_packingtrace_2M"

log("Simulating unified")
scheduler = Scheduler()
unified = scheduler.simulate()
unified["pools"] = "unified"
log(unified["pool_size"].describe())
unified.to_pickle(f"{filename}.unified.pkl")

log("Simulating fragmented")
scheduler = Scheduler(fragmented=True)
fragmented = scheduler.simulate()
fragmented["pools"] = "fragmented"
log(fragmented["pool_size"].describe())
fragmented.to_pickle(f"{filename}.fragmented.pkl")

df = pd.concat([unified, fragmented])


sns.lineplot(
    data=df,
    x="time",
    y="pool_size",
    hue="pools",
    style="pools",
    # label=f'{self._name}',
    # color=self._line_color,
    # linestyle=self._line,
)

df.to_pickle(f"{filename}.pkl")
plt.savefig(f"{filename}.pdf")
