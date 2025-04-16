import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Dict
from dataclasses import dataclass

def log(msg):
    print(msg, flush=True)

# Connect to the database
conn = sqlite3.connect('AzurePackingTraceV1/packing_trace_zone_a_v1.sqlite')

# Load VM requests

log("Loading data")
vm_types = pd.read_sql_query("SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType", conn)

a = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, starttime as time_sorter FROM vm", conn)
b = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, endtime as time_sorter FROM vm", conn)
vm_requests = pd.concat([a, b]).sort_values("time_sorter") # one line/event for each start and end of a VM
vm_requests = vm_requests.head(10000)


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

    def start_vm(self, vm: VM, machine_type: pd.Series):
        self.vms.append(vm)
        self.core += machine_type["core"]
        self.memory += machine_type["memory"]
        self.hdd += machine_type["hdd"]
        self.ssd += machine_type["ssd"]
        self.nic += machine_type["nic"]

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
    vmId_to_machine_type: Dict[int, int] = dict()

    def schedule(self, vm_request: pd.Series):
        vm_type = vm_request["vmTypeId"]
        machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm_type]

        # TODO select optimal candidate
        optimal_type = machine_type_candidates.iloc[0]

        # bookkeeping
        self.machine_types[optimal_type["machineId"]] = self.machine_types.get(optimal_type["machineId"], [])
        machines = self.machine_types[optimal_type["machineId"]]
        vm = VM(vmId=vm_request["vmId"], vmTypeId=vm_request["vmTypeId"])
        if len(machines) == 0:
            machines.append(Machine(machineId=optimal_type["machineId"], vms=[]))
        machines[0].start_vm(vm, optimal_type)
        self.vmId_to_machine_type[vm.vmId] = optimal_type["machineId"]
        # machines[vm_request["vmTypeId"]]
        pass

    def unschedule(self, vm_request: pd.Series):
        vm_type = vm_request["vmTypeId"]
        vmId = vm_request["vmId"]
        machineId = self.vmId_to_machine_type[vmId]
        vm_usage = vm_types[(vm_types["vmTypeId"] == vm_type) & (vm_types["machineId"] == machineId)]
        assert len(vm_usage) == 1
        vm_usage = vm_usage.iloc[0]
        for machine in self.machine_types[machineId]:
            if machine.stop_vm(vmId, vm_usage):
                if len(machine.vms) == 0:
                    del self.machine_types[machineId]
                del self.vmId_to_machine_type[vmId]
                break

        pass


    def dump(self):
        print("Machine types:")
        for machineId, machines in self.machine_types.items():
            for machine in machines:
                print(f"Machine type {machineId}:")
                print(f"  {len(machine.vms)} VMs, core={machine.core}, memory={machine.memory}, hdd={machine.hdd}, ssd={machine.ssd}, nic={machine.nic}")


scheduler = Scheduler()

for index, row in tqdm(vm_requests.iterrows(), total=len(vm_requests)):
    if row["starttime"] == row["time_sorter"]:
        scheduler.schedule(row)
    elif row["endtime"] == row["time_sorter"]:
        scheduler.unschedule(row)
    else:
        assert False



breakpoint()
