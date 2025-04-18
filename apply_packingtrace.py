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
from enum import Enum
from math import nan as NaN

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
    parser.add_argument('-b',
                        '--bottlenecks',
                        action='store_true',
                        help='Measure bottlenecking resources (insanely slow)',
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


def rank_machines_by_network_speed(data):
    """
    Rank machines by inferred network speed based on network usage fractions.

    Parameters:
        data: List of tuples (vmTypeId, machineId, network_usage)

    Returns:
        List of tuples (machineId, relative_speed) sorted by speed (highest to lowest)
    """
    # Group by vmTypeId
    vm_to_machines = {}
    for vm_id, machine_id, usage in data:
        if vm_id not in vm_to_machines:
            vm_to_machines[vm_id] = []
        vm_to_machines[vm_id].append((machine_id, usage))

    # For each VM type, normalize and calculate relative speeds
    relative_speeds = {}
    for vm_id, machine_usages in vm_to_machines.items():
        # Find maximum usage within each VM type for normalization
        max_usage = max(usage for _, usage in machine_usages)

        for machine_id, usage in machine_usages:
            # Lower usage fraction implies higher network speed
            # Normalize to get relative speed compared to other machines
            normalized_speed = max_usage / usage if usage > 0 else float('inf')

            if machine_id not in relative_speeds:
                relative_speeds[machine_id] = []
            relative_speeds[machine_id].append(normalized_speed)

    # Calculate average relative speed for each machine
    avg_speeds = {}
    for machine_id, speeds in relative_speeds.items():
        avg_speeds[machine_id] = sum(speeds) / len(speeds)

    # Rank machines by average relative speed (highest first)
    ranked_machines = sorted(avg_speeds.items(), key=lambda x: x[1], reverse=True)

    return ranked_machines


def rank_machine_types(vm_types) -> pd.DataFrame:
    data = []
    for index, row in vm_types.iterrows():
        data += [(str(row["vmTypeId"]), str(row["machineId"]), float(row["nic"]))]

    # Example usage
    # data = [
    #     ("vm1", "machine1", 0.2),
    #     ("vm1", "machine2", 0.1),  # machine2 is faster for vm1 (uses half the fraction)
    #     ("vm2", "machine1", 0.3),
    #     ("vm2", "machine3", 0.15)  # machine3 is faster for vm2
    # ]

    ranked_machines = rank_machines_by_network_speed(data)
    print("Machines ranked by network speed (fastest to slowest):")
    for machine_id, speed in ranked_machines:
        print(f" - {machine_id} (Relative Speed: {speed:.2f})")

    relative_speeds = []
    for index, row in vm_types.iterrows():
        relative_speed = [ speed for machine_id, speed in ranked_machines if machine_id == str(row["machineId"]) ]
        assert len(relative_speed) == 1
        relative_speeds += [ float(relative_speed[0]) ]

    vm_types["nic_performance"] = relative_speeds

    return vm_types.sort_values("nic_performance", ascending=False, ignore_index=True)


def fragment_pool(vm_types: pd.DataFrame) -> pd.DataFrame:
    for vm_type in vm_types["vmTypeId"].unique():
        # vm_types.loc[vm_types["vmTypeId"] == vm_type, "machineId"] = vm_types.loc[vm_types["vmTypeId"] == vm_type, "machineId"].apply(lambda x: int(x) % 3)
        vm_types_n = vm_types[vm_types["vmTypeId"] == vm_type]
        num = len(vm_types_n)
    return vm_types


def load_data(sqlite_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)

    # Load VM requests

    log("Loading data")
    vm_types = pd.read_sql_query("SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType", conn)

    a = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, starttime as time_sorter FROM vm", conn)
    b = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, endtime as time_sorter FROM vm", conn)
    vm_requests = pd.concat([a, b]).sort_values("time_sorter", ignore_index=True) # one line/event for each start and end of a VM
    # vm_requests = vm_requests.head(100_000)

    return vm_requests, vm_types


class StartResult(Enum):
    Ok = 0
    CoreBottleneck = 1
    MemoryBottleneck = 2
    HddBottleneck = 3
    SsdBottleneck = 4
    NicBottleneck = 5


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

    def start_vm(self, vm: VM, machine_type: pd.Series) -> StartResult:
        _core = self.core + machine_type["core"]
        _memory = self.memory + machine_type["memory"]
        _hdd = self.hdd + machine_type["hdd"]
        _ssd = self.ssd + machine_type["ssd"]
        _nic = self.nic + machine_type["nic"]
        if _core > 1:
            return StartResult.CoreBottleneck
        if _memory > 1:
            return StartResult.MemoryBottleneck
        if _hdd > 1:
            return StartResult.HddBottleneck
        if _ssd > 1:
            return StartResult.SsdBottleneck
        if _nic > 1:
            return StartResult.NicBottleneck
        self.vms.append(vm)
        self.core = _core
        self.memory = _memory
        self.hdd = _hdd
        self.ssd = _ssd
        self.nic = _nic
        return StartResult.Ok

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
    checkpoint_basename: str

    machine_types: Dict[int, List[Machine]] = dict()
    vmId_to_machine: Dict[int, Tuple[int, int]] = dict() # vmId -> (machine_type, machine_idx)

    cores: float = 0
    memory: float = 0
    hdd: float = 0
    ssd: float = 0
    nic: float = 0

    find_bottlenecks: bool
    # for each scheduling event (list item), which fraction machines could not be used because of _ bottleneck
    core_bottleneck_f: List[float] = []
    memory_bottleneck_f: List[float] = []
    hdd_bottleneck_f: List[float] = []
    ssd_bottleneck_f: List[float] = []
    nic_bottleneck_f: List[float] = []

    fragmented: bool

    def __init__(self, fragmented: bool = False, find_bottlenecks: bool = False, checkpoint_basename: str = "/tmp/checkpoint"):
        self.fragmented = fragmented
        self.find_bottlenecks = find_bottlenecks
        self.checkpoint_basename = checkpoint_basename
        self.machine_types = dict()
        self.vmId_to_machine = dict()


    @staticmethod
    def restore(filename) -> Any:
        with open(filename, "rb") as f:
            return pickle.load(f)


    def checkpoint(self, output: pd.DataFrame | None = None):
        with open(f"{self.checkpoint_basename}.state.pkl", "wb") as f:
            pickle.dump(self, f)
        if output is not None:
            output.to_pickle(f"{self.checkpoint_basename}.output.pkl")


    def schedule(self, vm_request: pd.Series, vm_types):
        core_bottlenecks = 0
        memory_bottlenecks = 0
        hdd_bottlenecks = 0
        ssd_bottlenecks = 0
        nic_bottlenecks = 0

        vm_type = vm_request["vmTypeId"]
        machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm_type]

        # TODO select optimal candidate
        if self.fragmented:
            pool_fragment = int(vm_request["vmId"]) % 3
            options = len(machine_type_candidates)
            ranges = [
                0,
                # passthrough
                int(options / 3),
                # mediation
                int(options / 3 * 2),
                # emulation
                options
            ]
            # candidates in our fragmented pool are:
            # ranges[pool_fragment] ... ranges[pool_fragment + 1]
            optimal_type = machine_type_candidates.iloc[ranges[pool_fragment]]
        else:
            optimal_type = machine_type_candidates.iloc[0]

        # bookkeeping
        self.machine_types[optimal_type["machineId"]] = self.machine_types.get(optimal_type["machineId"], [])
        machines = self.machine_types[optimal_type["machineId"]]
        vm = VM(vmId=vm_request["vmId"], vmTypeId=vm_request["vmTypeId"])
        started = None
        machine_idx = None
        if self.find_bottlenecks:
            iterator = range(len(machines)) # forwards
            # TODO we also need to remove the `break` from the loop below, so that we
            # check bottlenecks on all machines (instead of only the ones we try until
            # we find a free one).
        else:
            iterator = range(len(machines) - 1, 0, -1) # backwards
        for i in iterator:
            machine = machines[i]
            started = machine.start_vm(vm, optimal_type)
            if self.find_bottlenecks:
                if started == StartResult.CoreBottleneck:
                    core_bottlenecks += 1
                if started == StartResult.MemoryBottleneck:
                    memory_bottlenecks += 1
                if started == StartResult.HddBottleneck:
                    hdd_bottlenecks += 1
                if started == StartResult.SsdBottleneck:
                    ssd_bottlenecks += 1
                if started == StartResult.NicBottleneck:
                    nic_bottlenecks += 1
            if started == StartResult.Ok:
                machine_idx = i
                break

        # create new machine if necessary
        if started is None or started != StartResult.Ok:
            machines.append(Machine(machineId=optimal_type["machineId"], vms=[]))
            started = machines[-1].start_vm(vm, optimal_type)
            machine_idx = len(machines) - 1
        assert started == StartResult.Ok

        # collect bottleneck statistics
        if self.find_bottlenecks:
            total_bottlenecks = core_bottlenecks + memory_bottlenecks + hdd_bottlenecks + ssd_bottlenecks + nic_bottlenecks
            if total_bottlenecks > 0:
                self.core_bottleneck_f += [core_bottlenecks / total_bottlenecks]
                self.memory_bottleneck_f += [memory_bottlenecks / total_bottlenecks]
                self.hdd_bottleneck_f += [hdd_bottlenecks / total_bottlenecks]
                self.ssd_bottleneck_f += [ssd_bottlenecks / total_bottlenecks]
                self.nic_bottleneck_f += [nic_bottlenecks / total_bottlenecks]
            else:
                self.core_bottleneck_f += [0]
                self.memory_bottleneck_f += [0]
                self.hdd_bottleneck_f += [0]
                self.ssd_bottleneck_f += [0]
                self.nic_bottleneck_f += [0]

        self.cores += optimal_type["core"]
        self.memory += optimal_type["memory"]
        self.hdd += optimal_type["hdd"]
        self.ssd += optimal_type["ssd"]
        self.nic += optimal_type["nic"]

        self.vmId_to_machine[vm.vmId] = (optimal_type["machineId"], machine_idx)
        # machines[vm_request["vmTypeId"]]


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


    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, checkpoint_interval: int | None = 100_000) -> pd.DataFrame:
        time = None
        time_series_time = []
        time_series_pool_size = []
        time_series_cores = []
        time_series_memory = []
        time_series_hdd = []
        time_series_ssd = []
        time_series_nic = []
        time_series_core_bottleneck = []
        time_series_memory_bottleneck = []
        time_series_hdd_bottleneck = []
        time_series_ssd_bottleneck = []
        time_series_nic_bottleneck = []

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
                df = pd.DataFrame({
                    "time": time_series_time,
                    "pool_size": time_series_pool_size,
                    "cores": time_series_cores,
                    "memory": time_series_memory,
                    "hdd": time_series_hdd,
                    "ssd": time_series_ssd,
                    "nic": time_series_nic,
                    "core_bottleneck": time_series_core_bottleneck,
                    "memory_bottleneck": time_series_memory_bottleneck,
                    "hdd_bottleneck": time_series_hdd_bottleneck,
                    "ssd_bottleneck": time_series_ssd_bottleneck,
                    "nic_bottleneck": time_series_nic_bottleneck,
                })
                self.checkpoint(output=df)

            if row["time_sorter"] is None:
                continue
            if time is None:
                time = row["time_sorter"]
                continue
            if row["time_sorter"] > time:
                # we have processed all events for this current point in time
                # (multiple rows can have the same time for which we aggregate results)
                time = row["time_sorter"]
                time_series_time += [time]
                time_series_pool_size += [self.pool_size()]
                # cluster usage (1-stranding) statistics
                time_series_cores += [ self.cores ]
                time_series_memory += [ self.memory ]
                time_series_hdd += [ self.hdd ]
                time_series_ssd += [ self.ssd ]
                time_series_nic += [ self.nic ]
                # scheduler bottleneck statistics
                if self.find_bottlenecks:
                    time_series_core_bottleneck += [sum(self.core_bottleneck_f) / len(self.core_bottleneck_f)]
                    time_series_memory_bottleneck += [sum(self.memory_bottleneck_f) / len(self.memory_bottleneck_f)]
                    time_series_hdd_bottleneck += [sum(self.hdd_bottleneck_f) / len(self.hdd_bottleneck_f)]
                    time_series_ssd_bottleneck += [sum(self.ssd_bottleneck_f) / len(self.ssd_bottleneck_f)]
                    time_series_nic_bottleneck += [sum(self.nic_bottleneck_f) / len(self.nic_bottleneck_f)]
                    self.core_bottleneck_f.clear()
                    self.memory_bottleneck_f.clear()
                    self.hdd_bottleneck_f.clear()
                    self.ssd_bottleneck_f.clear()
                    self.nic_bottleneck_f.clear()
                else:
                    time_series_core_bottleneck += [NaN]
                    time_series_memory_bottleneck += [NaN]
                    time_series_hdd_bottleneck += [NaN]
                    time_series_ssd_bottleneck += [NaN]
                    time_series_nic_bottleneck += [NaN]

        df = pd.DataFrame({
            "time": time_series_time,
            "pool_size": time_series_pool_size,
            "cores": time_series_cores,
            "memory": time_series_memory,
            "hdd": time_series_hdd,
            "ssd": time_series_ssd,
            "nic": time_series_nic,
            "core_bottleneck": time_series_core_bottleneck,
            "memory_bottleneck": time_series_memory_bottleneck,
            "hdd_bottleneck": time_series_hdd_bottleneck,
            "ssd_bottleneck": time_series_ssd_bottleneck,
            "nic_bottleneck": time_series_nic_bottleneck,
            "fragmented": self.fragmented,
        })
        return df


def main():
    parser = setup_parser()
    args = parse_args(parser)
    print(args)

    vm_requests, vm_types = load_data(args.input)

    log("Ranking NICs")
    vm_types = rank_machine_types(vm_types)

    log("Simulating")
    checkpoint_basename = f"{args.output}.checkpoint"
    scheduler = Scheduler(fragmented=args.fragmented, find_bottlenecks=args.bottlenecks, checkpoint_basename=checkpoint_basename)
    if args.restore is not None:
        scheduler.restore(args.restore)
        scheduler.checkpoint_basename = checkpoint_basename

    df = scheduler.simulate(vm_requests, vm_types)

    scheduler.checkpoint(output=df)
    log(df["pool_size"].describe())
    df.to_pickle(f"{args.output}.pkl")
    log(f"Wrote output to {args.output}.pkl")


if __name__ == "__main__":
    main()
