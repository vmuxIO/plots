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
from ortools.linear_solver import pywraplp
import time
import cProfile
import pstats
import io
import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty as QueueEmpty
import ctypes

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
    parser.add_argument('-p',
                        '--optimal',
                        action='store_true',
                        help='Use optimal scheduler',
                        )
    parser.add_argument('-m',
                        '--migrate',
                        action='store_true',
                        help='Consider VM migration while scheduling',
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
    # print("Machines ranked by network speed (fastest to slowest):")
    # for machine_id, speed in ranked_machines:
    #     print(f" - {machine_id} (Relative Speed: {speed:.2f})")

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


def load_data(sqlite_file: str, no_timeline: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Connect to the database
    conn = sqlite3.connect(sqlite_file)

    # Load VM requests

    log("Loading data")
    vm_types = pd.read_sql_query("SELECT vmTypeId, machineId, core, memory, hdd, ssd, nic FROM vmType", conn)

    a = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, starttime as time_sorter FROM vm", conn)
    if not no_timeline:
        b = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime, endtime as time_sorter FROM vm", conn)
        vm_requests = pd.concat([a, b]).sort_values("time_sorter", ignore_index=True) # one line/event for each start and end of a VM
    else:
        vm_requests = a.sort_values("time_sorter", ignore_index=True) # for consistency
    # vm_requests = vm_requests.head(100_000)

    return vm_requests, vm_types


def adaptive_sampler(start=0, end=14):
    """
    Creates a sampler that progressively refines the sampling resolution
    while maintaining roughly equal spacing at any point if aborted.

    Args:
        start (float): The starting value of the range
        end (float): The ending value of the range

    Returns:
        function: A function that returns the next sample when called
    """
    sampled_points = set()
    # Start with endpoints and midpoint in the queue
    points_to_sample = [start, end, (start + end) / 2]

    def get_next_sample():
        if not points_to_sample:
            # Find the largest gap and add its midpoint
            all_points = sorted(list(sampled_points))

            if len(all_points) <= 1:
                return None  # No more points to sample

            # Find the largest gap
            largest_gap = 0
            gap_midpoint = None

            for i in range(len(all_points) - 1):
                gap = all_points[i+1] - all_points[i]
                if gap > largest_gap:
                    largest_gap = gap
                    gap_midpoint = all_points[i] + gap / 2

            if gap_midpoint is not None:
                points_to_sample.append(gap_midpoint)

        if not points_to_sample:
            return None

        next_point = points_to_sample.pop(0)
        sampled_points.add(next_point)
        return next_point

    return get_next_sample


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
        return self.start_vm2(vm,
                              machine_type["core"],
                              machine_type["memory"],
                              machine_type["hdd"],
                              machine_type["ssd"],
                              machine_type["nic"]
        )

    def start_vm2(self, vm: VM, core: float, memory: float, hdd: float, ssd: float, nic: float) -> StartResult:
        _core = self.core + core
        _memory = self.memory + memory
        _hdd = self.hdd + hdd
        _ssd = self.ssd + ssd
        _nic = self.nic + nic
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

        core = float(optimal_type["core"])
        memory = float(optimal_type["memory"])
        hdd = float(optimal_type["hdd"])
        ssd = float(optimal_type["ssd"])
        nic = float(optimal_type["nic"])

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
            started = machine.start_vm2(vm, core, memory, hdd, ssd, nic)
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
            started = machines[-1].start_vm2(vm, core, memory, hdd, ssd, nic)
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
        print(f"{self.pool_size()} Machines:")
        for machineId, machines in self.machine_types.items():
            for machine in machines:
                print(f" - Machine type {machineId}:")
                print(f"   {len(machine.vms)} VMs, core={machine.core}, memory={machine.memory}, hdd={machine.hdd}, ssd={machine.ssd}, nic={machine.nic}")


    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, checkpoint_interval: int | None = 100_000) -> pd.DataFrame:
        # vm_requests = vm_requests[vm_requests["time_sorter"] < -800]
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


class FakeFile():
    def __init__(self, shared_string):
        self.buffer = "<empty>"
        self.shared_string = shared_string

    def write(self, msg: str):
        self.buffer = msg

    def flush(self):
        printable = ''.join(i for i in self.buffer if i.isprintable())
        if len(printable) < 5:
            return # ignore terminal control characters
        self.shared_string.value = printable.encode("utf-8")
        # log(f"flush")
        # log(f"1 {self.buffer}")
        # log(f"2 {self.shared_string.value}")
        # self.shared_string.value = b"<foobar>"


class Progress():
    def __init__(self, shared_string):
        self.shared_string = shared_string

    def tqdm(self, iterable: Any, *args, **kwargs) -> Any:
        return tqdm(iterable, *args, file=FakeFile(self.shared_string), ncols=50, mininterval=1, ascii=False, **kwargs)


class ProgressCollector():
    def __init__(self):
        self.shared_strings = dict()

    def new_progress(self, process: int) -> Any:
        default_string = mp.Array('c', b"<Proc {process}                                                                                                                                                                          >")
        self.shared_strings[process] = default_string
        return default_string

    def print(self):
        lines = ""
        for process, shared_string in self.shared_strings.items():
            # log(shared_string.value)
            string = shared_string.value.decode('utf-8')
            lines += f"P{process:3d} {string} "
            if (process+1) % 3 == 0:
                lines += "\n"
        with open("/tmp/progress", "w") as f:
            f.write(lines)

    def test(self):
        # for i in tqdm(range(3), total=3, file=FakeFile(), ncols=50, mininterval=1, ascii=False):
        progress = Progress()
        for i in progress.tqdm(range(3), total=3):
            time.sleep(1)
            pass



class MigratingScheduler(Scheduler):
    def __init__(self, progress: Progress | None = None, fragmented: bool = False, find_bottlenecks: bool = False, checkpoint_basename: str = "/tmp/checkpoint"):
        self.progress = progress
        super().__init__(fragmented=fragmented, find_bottlenecks=find_bottlenecks, checkpoint_basename=checkpoint_basename)


    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, checkpoint_interval: int | None = None) -> pd.DataFrame:
        samples = []
        samples_queued = 0
        # samples += [ self.sample(vm_requests, vm_types, -7157) ]
        # samples += [ self.sample(vm_requests, vm_types, 1) ]

        num_cores = mp.cpu_count()
        # num_cores = 4
        task_queue = Queue()
        result_queue = Queue()
        progress = ProgressCollector()
        next_time_sample = adaptive_sampler(0, 14)
        max_samples = 150
        # max_samples = 14 * 24 * 60 # once a minute for 14 days
        # next_time_sample = adaptive_sampler(-1400, -1200)
        # max_samples = 10
        progress_bar = tqdm(range(max_samples))


        def process_task(task_queue, result_queue, progress_string):
            """Worker function to process tasks from the queue"""
            progress = Progress(progress_string)
            proc_name = mp.current_process().name
            while True:
                # Get a task from the queue
                sample_time = task_queue.get()

                # Check for termination signal
                if sample_time is None:
                    print(f"{proc_name} exiting")
                    task_queue.put(None)  # Help other workers terminate
                    break

                # Process the task
                print(f"{proc_name} processing time: {sample_time}")
                # each task needs to use its own fresh Scheduler state
                scheduler = MigratingScheduler(progress=progress, fragmented=self.fragmented, find_bottlenecks=self.find_bottlenecks, checkpoint_basename=self.checkpoint_basename)
                result = scheduler.sample(vm_requests, vm_types, sample_time)
                # result = 1
                # time.sleep(5)
                result_queue.put((sample_time, result))

        # Create and start the worker processes
        processes = []
        for i in range(num_cores):
            p = mp.Process(target=process_task, args=(task_queue, result_queue, progress.new_progress(process=i)))
            p.start()
            processes.append(p)

        # Add initial tasks to queue
        for i in range(2*num_cores):
            task_queue.put(next_time_sample())
            samples_queued += 1

        # Add more tasks until we have enough samples
        while len(samples) < max_samples:
            try:
                task, result = result_queue.get(timeout=1)
            except QueueEmpty:
                progress.print()
            else:
                progress.print()
                samples += [ result ]
                if samples_queued < max_samples:
                    task_queue.put(next_time_sample())
                    samples_queued += 1
                df = pd.concat(samples)
                df.to_pickle(f"{self.checkpoint_basename}.samples.pkl")
                log(f"Written {len(samples)} samples to {self.checkpoint_basename}.samples.pkl")
                progress_bar.update(1)


        # Add termination sentinel so that processes can exit nicely
        task_queue.put(None)

        # Haha, just kidding
        for p in processes:
            p.terminate()

        return pd.concat(samples)


    def sample(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, time: int) -> pd.DataFrame:
        # log(f"Sample at time {time}")
        active_mask = (vm_requests['starttime'] <= time) & ((vm_requests['endtime'].isnull()) | (vm_requests['endtime'] >= time))
        active_vms = vm_requests[active_mask]
        # log(f"Active VMs: {len(active_vms)}")

        # active_vms = vm_requests

        df = self.solve(active_vms, vm_types)
        df["time"] = time

        return df


    def solve(self, active_vms: pd.DataFrame, vm_types: pd.DataFrame) -> pd.DataFrame:
        # sort VMs by size. Place the biggest ones first.

        canidate_types = vm_types.drop_duplicates(subset="vmTypeId", keep="first")
        join = pd.merge(active_vms, canidate_types, on="vmTypeId", how="left")
        # sorted = join.sort_values(["core", "memory", "nic"], ascending=False, ignore_index=True)
        sorted = join

        return self._simulate(sorted)


    def _simulate(self, sorted_vms: pd.DataFrame) -> pd.DataFrame:
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

        # for index, row in tqdm(vm_requests.iterrows(), total=len(vm_requests)):
        if self.progress is None:
            iter = tqdm(sorted_vms.iterrows(), total=len(sorted_vms))
        else:
            iter = self.progress.tqdm(sorted_vms.iterrows(), total=len(sorted_vms))
        for index, row in iter:
            # if (int(index)%1000) == 0:
            #     pr = cProfile.Profile()
            #     pr.enable()
            self.schedule2(row)
            # if (int(index)%1000) == 0:
            #     pr.disable()
            #     s = io.StringIO()
            #     sortby = pstats.SortKey.CUMULATIVE
            #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #     ps.print_stats()
            #     print(s.getvalue())

            # if checkpoint_interval is not None and int(index) % checkpoint_interval == 0:
            #     self.iteration = int(index)
            #     df = pd.DataFrame({
            #         "time": time_series_time,
            #         "pool_size": time_series_pool_size,
            #         "cores": time_series_cores,
            #         "memory": time_series_memory,
            #         "hdd": time_series_hdd,
            #         "ssd": time_series_ssd,
            #         "nic": time_series_nic,
            #         "core_bottleneck": time_series_core_bottleneck,
            #         "memory_bottleneck": time_series_memory_bottleneck,
            #         "hdd_bottleneck": time_series_hdd_bottleneck,
            #         "ssd_bottleneck": time_series_ssd_bottleneck,
            #         "nic_bottleneck": time_series_nic_bottleneck,
            #     })
            #     self.checkpoint(output=df)

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
            # "time": time_series_time,
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


    def schedule2(self, joined_vm: pd.Series):
        core_bottlenecks = 0
        memory_bottlenecks = 0
        hdd_bottlenecks = 0
        ssd_bottlenecks = 0
        nic_bottlenecks = 0

        machineType = float(joined_vm["machineId"])
        core = float(joined_vm["core"])
        memory = float(joined_vm["memory"])
        hdd = float(joined_vm["hdd"])
        ssd = float(joined_vm["ssd"])
        nic = float(joined_vm["nic"])

        # bookkeeping
        self.machine_types[machineType] = self.machine_types.get(machineType, [])
        machines = self.machine_types[machineType]
        vm = VM(vmId=joined_vm["vmId"], vmTypeId=joined_vm["vmTypeId"])
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
            started = machine.start_vm2(vm, core, memory, hdd, ssd, nic)
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
            machines.append(Machine(machineId=machineType, vms=[]))
            started = machines[-1].start_vm2(vm, core, memory, hdd, ssd, nic)
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

        self.cores += joined_vm["core"]
        self.memory += joined_vm["memory"]
        self.hdd += joined_vm["hdd"]
        self.ssd += joined_vm["ssd"]
        self.nic += joined_vm["nic"]

        self.vmId_to_machine[vm.vmId] = (machineType, machine_idx)
        # machines[vm_request["vmTypeId"]]


class OptimalScheduler2():
    def __init__(self):
        pass

    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame):
        # useful resource: https://github.com/AKafakA/vm_schedule_simulator/blob/main/algorithm/schedule/optimal_bin_packing.py

        # see https://developers.google.com/optimization/lp/lp_advanced for ortools solvers (e.g. GLOP, Linerar optimization)
        # https://developers.google.com/optimization/mip/mip_example for a mention of SCIP (Integer optimization)
        # perhaps we could also consider bin-packing specific optimizers https://developers.google.com/optimization/pack/bin_packing
        solver = pywraplp.Solver.CreateSolver("SAT")
        assert solver is not None

        workerIds = [ 0, 1 ]
        vm_resources = {
            # resources to be allocated for a VM
            0: 400,
            1: 500,
        }

        # x[i, b] = 1 if VM i is packed in machine b.
        x = {}
        x[0, 0] = solver.BoolVar("x_0_0") # 0
        x[0, 1] = solver.BoolVar("x_0_1") # 1
        x[1, 1] = solver.BoolVar("x_0_1") # 1

        # Assume one task only be placed into one machines
        # TODO maybe == 1 to force packing of all VMs?
        solver.Add(sum(x[0, b] for b in [ 0, 1 ]) == 1)
        solver.Add(sum(x[1, b] for b in [ 1 ]) == 1)

        # The amount packed in each bin cannot exceed its capacity (1).
        solver.Add( # VMs that can run on worker 0: [ 0 ]
            sum(x[i, 0] * vm_resources[i] for i in [ 0 ])
            <= 1000
        )
        solver.Add( # VMs that can run on worker 1: [ 0, 1 ]
            sum(x[i, 1] * vm_resources[i] for i in [ 0, 1 ])
            <= 1000
        )


        # m[j] = 1 if machine hosts any VM
        m = {}
        m[0] = solver.BoolVar("m_0")
        solver.Add(m[0] ==
                   sum([x[0, 0]])
        )
        m[1] = solver.BoolVar("m_1")
        solver.Add(m[1] ==
                   sum([x[0, 1], x[1, 1]])
        )
        # sum(x) - (sum(x) - 1) == 0 | 1
        # sum(x) - sum(x) + 1) == 0 | 1

        # # even our trivial scheduler needs less than 150k machines
        machines = solver.IntVar(0, 5, "machines")
        solver.Add(machines == m[0] + m[1])

        utilization = solver.IntVar(0, 5, "utilization")
        solver.Add(utilization == sum([ x[0, 0] * vm_resources[0], x[0, 1] * vm_resources[0], x[1, 1] * vm_resources[1] ]) )

        # objective = solver.Objective()

        #  Objective is to maximize the cpu resources usage
        # objective.SetCoefficient(x[0, 0], vm_resources[0])
        # objective.SetCoefficient(x[0, 1], vm_resources[0])
        # objective.SetCoefficient(x[1, 1], vm_resources[1])
        # objective.SetMaximization()

        #  Objective is to minimize number of machines
        # objective.SetCoefficient(m[0], 1)
        # objective.SetCoefficient(m[1], 1)
        # objective.SetMinimization()
        # objective.SetCoefficient(totalMachines, 1)

        solver.Minimize(machines)
        # solver.Maximize(utilization)

        status = solver.Solve()
        print(f"solution: {status}")
        if status == pywraplp.Solver.OPTIMAL:
            print(f"x_0_0 {x[0, 0].solution_value()}")
            print(f"x_0_1 {x[0, 1].solution_value()}")
            print(f"x_1_1 {x[1, 1].solution_value()}")
            # print(f"m_0 {m[0].solution_value()}")
            # print(f"m_1 {m[1].solution_value()}")
            print(f"util {utilization.solution_value()}")
            print(f"machines {machines.solution_value()}")

        breakpoint()
        pass


class OptimalScheduler():
    def __init__(self):
        pass

    def simulate(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame):
        self.sample(vm_requests, vm_types, -7157)
        pass

    def sample(self, vm_requests: pd.DataFrame, vm_types: pd.DataFrame, time: int):
        # active_mask = (vm_requests['starttime'] <= time) & ((vm_requests['endtime'].isnull()) | (vm_requests['endtime'] >= time))
        # active_vms = vm_requests[active_mask]

        active_vms = vm_requests

        self.solve(active_vms, vm_types)

        pass

    def solve(self, active_vms: pd.DataFrame, vm_types: pd.DataFrame):
        start = time.time()
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")
        assert solver is not None

        max_instances_per_type = 20
        machine_capacity = 1

        # Variables

        log("Creating x_i_t_j")
        # x[i, t, j] = 1 if VM i is packed in machine j of machine type t.
        x = {}
        for _vm_idx, vm in tqdm(active_vms.iterrows(), total=len(active_vms)):
            i = vm["vmId"]
            machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm["vmTypeId"]]
            for _type_idx, machine_type_candidate in machine_type_candidates.iterrows():
                t = machine_type_candidate["machineId"]
                for j in range(max_instances_per_type):
                    x[(i, t, j)] = solver.IntVar(0, 1, f"x_{i}_{t}_{j}")
        log(f"  len(x_i_t_j) = {len(x.values())}")

        log("Creating y_t_j")
        # y[t, j] = 1 if machine j of type t is used.
        y = {}
        for _type_idx, machine_types in vm_types.iterrows():
            t = machine_types["machineId"]
            for j in range(max_instances_per_type):
                y[t, j] = solver.IntVar(0, 1, f"y_{t}_{j}")
        log(f"  len(y_t_j) = {len(y.values())}")

        # Constraints

        # Each VM must be in exactly one machine.
        log("Constraining for all VMs i, sum(x_i_t_j) == 1")
        for _vm_idx, vm in active_vms.iterrows():
            relevant_x = [] # all x[i, t, j] for this VM

            i = vm["vmId"]
            machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm["vmTypeId"]]
            for _type_idx, machine_type_candidate in machine_type_candidates.iterrows():
                t = machine_type_candidate["machineId"]
                for j in range(max_instances_per_type):
                    relevant_x += [x[i, t, j]]

            solver.Add(sum(relevant_x) == 1)

        # The amount packed in each bin cannot exceed its capacity.
        log("Constraining for all VMs i, sum(x_i_t_j * core_i_t) <= y_t_j * machine_capacity")
        for _vm_idx, vm in tqdm(active_vms.iterrows(), total=len(active_vms)):
            relevant_core_sum = []
            relevant_memory_sum = []
            relevant_ssd_sum = []
            relevant_nic_sum = []
            # we ignore hdd because it matters so little, the data is even missing often

            # collect sum(*) into relevant_core_sum
            i = vm["vmId"]
            machine_type_candidates = vm_types[vm_types["vmTypeId"] == vm["vmTypeId"]]
            for _type_idx, machine_type_candidate in machine_type_candidates.iterrows():
                t = machine_type_candidate["machineId"]
                for j in range(max_instances_per_type):
                    relevant_core_sum += [ x[i, t, j] * machine_type_candidate["core"] ]
                    relevant_memory_sum += [ x[i, t, j] * machine_type_candidate["memory"] ]
                    relevant_ssd_sum += [ x[i, t, j] * machine_type_candidate["ssd"] ]
                    relevant_nic_sum += [ x[i, t, j] * machine_type_candidate["nic"] ]

            # add contstraints using sum(*)
            for _type_idx, machine_type_candidate in machine_type_candidates.iterrows():
                t = machine_type_candidate["machineId"]
                for j in range(max_instances_per_type):
                    solver.Add(
                        sum(relevant_core_sum)
                        <= y[t, j] * machine_capacity
                    )
                    solver.Add(
                        sum(relevant_memory_sum)
                        <= y[t, j] * machine_capacity
                    )
                    solver.Add(
                        sum(relevant_ssd_sum)
                        <= y[t, j] * machine_capacity
                    )
                    solver.Add(
                        sum(relevant_nic_sum)
                        <= y[t, j] * machine_capacity
                    )

        # Objective: minimize the number of machines used.
        # minimize: sum(y_t_j)
        solver.Minimize(solver.Sum(y.values()))

        start_solve = time.time()
        print(f"Solving with {solver.SolverVersion()}")
        status = solver.Solve()

        match status:
            case pywraplp.Solver.OPTIMAL:
                print("Optimal solution found")
            case pywraplp.Solver.FEASIBLE:
                print("Solution feasible, or stopped by limit")
            case pywraplp.Solver.INFEASIBLE:
                print("Proven infeasible")
            case pywraplp.Solver.UNBOUNDED:
                print("Proven unbounded")
            case pywraplp.Solver.ABNORMAL:
                print("Abnormal error of some kind")
            case pywraplp.Solver.NOT_SOLVED:
                print("Not solved")
            case _:
                assert False, "Unknown solver status"

        if status == pywraplp.Solver.OPTIMAL:
            print("Machines:")
            num_machines = 0
            num_vms = 0
            for (t, j), y_t_j in y.items():
                if y_t_j.solution_value() == 1:
                    vms = []
                    core_usage = 0
                    memory_usage = 0
                    ssd_usage = 0
                    nic_usage = 0
                    # for (i, t, j), x_i_t_j in x.items():
                    for _vm_idx, vm in active_vms.iterrows():
                        i = vm["vmId"]
                        if (i, t, j) in x and x[i, t, j].solution_value() == 1:
                            vms += [ i ]
                            core_usage += float(vm_types[(vm_types["vmTypeId"] == vm["vmTypeId"]) & (vm_types["machineId"] == t)]["core"])
                            memory_usage += float(vm_types[(vm_types["vmTypeId"] == vm["vmTypeId"]) & (vm_types["machineId"] == t)]["memory"])
                            ssd_usage += float(vm_types[(vm_types["vmTypeId"] == vm["vmTypeId"]) & (vm_types["machineId"] == t)]["ssd"])
                            nic_usage += float(vm_types[(vm_types["vmTypeId"] == vm["vmTypeId"]) & (vm_types["machineId"] == t)]["nic"])
                    if vms:
                        num_machines += 1
                        num_vms += len(vms)
                        print(f" - Machine {j} of type {t}, {len(vms)} VMs, core {core_usage:.2f}, memory {memory_usage:.2f}, ssd {ssd_usage:.2f}, nic {nic_usage:.2f}")
            print(f"Number of machines used: {num_machines}")
            print(f"Number of VMs: {num_vms}")

            # print("VMs:")
            # num_vms = 0
            # for (i, t, j), x_i_t_j in x.items():
            #     if x_i_t_j.solution_value() == 1:
            #         print(f" - VM {i} on machine {j} of type {t}")
            #         num_vms += 1
            # print(f"VMs total: {num_vms}")

        log(f"{(start_solve - start)/60:.1f} minutes to setup, {(time.time() - start_solve)/60:.1f} minutes to solve")

        breakpoint()
        pass


def main():
    start = time.time()
    parser = setup_parser()
    args = parse_args(parser)
    print(args)

    if args.migrate or args.optimal:
        vm_requests, vm_types = load_data(args.input, no_timeline=True)
    else:
        vm_requests, vm_types = load_data(args.input)
    # vm_requests = vm_requests.head(1000)

    log("Ranking NICs")
    vm_types = rank_machine_types(vm_types)

    log("Simulating")
    checkpoint_basename = f"{args.output}.checkpoint"
    if args.optimal:
        scheduler = OptimalScheduler()
    elif args.migrate:
        scheduler = MigratingScheduler(fragmented=args.fragmented, find_bottlenecks=args.bottlenecks, checkpoint_basename=checkpoint_basename)
    else:
        scheduler = Scheduler(fragmented=args.fragmented, find_bottlenecks=args.bottlenecks, checkpoint_basename=checkpoint_basename)
    if args.restore is not None:
        scheduler.restore(args.restore)
        scheduler.checkpoint_basename = checkpoint_basename

    df = scheduler.simulate(vm_requests, vm_types)

    scheduler.checkpoint(output=df)
    log(df["pool_size"].describe())
    scheduler.dump()
    df.to_pickle(f"{args.output}.pkl")
    log(f"Wrote output to {args.output}.pkl")
    log(f"Took {(time.time() - start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
