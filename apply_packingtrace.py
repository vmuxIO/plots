import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple, Self
from dataclasses import dataclass
import pickle
import argparse

def log(msg):
    print(msg, flush=True)

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Simulate Azure Packing trace'
    )

    parser.add_argument('-f',
                        '--fragmented',
                        action='store_true',
                        help='Use fragmented pools',
                        )
    # parser.add_argument('-W', '--width',
    #                     type=float,
    #                     default=12,
    #                     help='Width of the plot in inches'
    #                     )
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default="/tmp/plot_packingtrace",
                        help='Basename for the output files',
                        )
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        default="packing_trace_zone_a_v1.sqlite",
                        help='Sqlite file of Azure Packing Trace v1',
                        )
    parser.add_argument('-r',
                        '--restore',
                        type=str,
                        help='Checkpoint to continue from',
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


def load_data(sqlite_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)

    # Load VM requests

    log("Loading data")
    vm_types = pd.read_sql_query("SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType", conn)

    a = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, starttime as time_sorter FROM vm", conn)
    b = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, endtime as time_sorter FROM vm", conn)
    vm_requests = pd.concat([a, b]).sort_values("time_sorter", ignore_index=True) # one line/event for each start and end of a VM
    vm_requests = vm_requests.head(2_000_000)

    return vm_requests, vm_types


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
    iteration: int = 0 # update when checkpointing
    checkpoint_file: str

    machine_types: Dict[int, List[Machine]] = dict()
    vmId_to_machine: Dict[int, Tuple[int, int]] = dict() # vmId -> (machine_type, machine_idx)

    cores: float = 0
    memory: float = 0
    hdd: float = 0
    ssd: float = 0
    nic: float = 0

    fragmented: bool

    def __init__(self, fragmented: bool = False, checkpoint_file: str = "/tmp/checkpoint.pkl"):
        self.fragmented = fragmented
        self.checkpoint_file = checkpoint_file
        self.machine_types = dict()
        self.vmId_to_machine = dict()


    @staticmethod
    def restore(filename) -> Any:
        with open(filename, "rb") as f:
            return pickle.load(f)


    def checkpoint(self):
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(self, f)


    def schedule(self, vm_request: pd.Series, vm_types):
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

    def unschedule(self, vm_request: pd.Series, vm_types):
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


    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, checkpoint_interval: int | None = 1_000) -> pd.DataFrame:
        time = None
        time_series_time = []
        time_series_pool_size = []

        for index, row in tqdm(vm_requests.iterrows(), total=len(vm_requests)):
            if int(index) < self.iteration:
                continue # skip requests that we already processed (e.g. loaded checkpoint)

            if row["starttime"] == row["time_sorter"]:
                self.schedule(row, vm_types)
            elif row["endtime"] == row["time_sorter"]:
                self.unschedule(row, vm_types)
            else:
                assert False

            if checkpoint_interval is not None and int(index) % checkpoint_interval == 0:
                self.iteration = int(index)
                self.checkpoint()

            if row["time_sorter"] is None:
                continue
            if time is None:
                time = row["time_sorter"]
                continue
            if row["time_sorter"] > time:
                time = row["time_sorter"]
                time_series_time += [time]
                time_series_pool_size += [self.pool_size()]

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


def main():
    parser = setup_parser()
    args = parse_args(parser)
    print(args)

    vm_requests, vm_types = load_data(args.input)

    log("Simulating")
    checkpoint_file = f"{args.output}.checkpoint.pkl"
    scheduler = Scheduler(fragmented=args.fragmented, checkpoint_file=checkpoint_file)
    if args.restore is not None:
        scheduler.restore(args.restore)
        scheduler.checkpoint_file = checkpoint_file

    df = scheduler.simulate(vm_requests, vm_types)

    scheduler.checkpoint()
    df["fragmented"] = args.fragmented
    log(df["pool_size"].describe())
    df.to_pickle(f"{args.output}.pkl")


if __name__ == "__main__":
    main()
