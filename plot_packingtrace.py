import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass
import argparse

def log(msg):
    print(msg, flush=True)

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot packet loss graph'
    )

    parser.add_argument('-t',
                        '--title',
                        type=str,
                        help='Title of the plot',
                        )
    parser.add_argument('-W', '--width',
                        type=float,
                        default=12,
                        help='Width of the plot in inches'
                        )
    parser.add_argument('-H', '--height',
                        type=float,
                        default=6,
                        help='Height of the plot in inches'
                        )
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help='''Path to the output plot
                             (default: packet_loss.pdf)''',
                        default='ycsb.pdf'
                        )
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


parser = setup_parser()
args = parse_args(parser)

sns.set_theme()
sns.set_style("whitegrid")

fig = plt.figure(figsize=(args.width, args.height))


df = pd.read_pickle("/tmp/packingtrace_0.1M_unified.pkl")

# df = df[df["time"] >= 0]


breakpoint()

log("plot packingtrace data")
sns.lineplot(
    data=df,
    x="time",
    y="pool_size",
    hue="fragmented",
    style="fragmented",
    # label=f'{self._name}',
    # color=self._line_color,
    # linestyle=self._line,
)
plt.savefig("packingtrace.pdf")

fig = plt.figure(figsize=(args.width, args.height))

log("cruching utilization data")
dfs = []
for resource in ["cores", "memory", "hdd", "ssd", "nic"]:
    d = dict()
    d["time"] = df["time"]
    d["utilization"] = df[resource] / df["pool_size"]
    d["stranding"] = 1 - (df[resource] / df["pool_size"])
    d["resource"] = resource
    dfs += [ pd.DataFrame(d) ]
df = pd.concat(dfs, ignore_index=True)

breakpoint()

log("plot utilization data")
g = sns.lineplot(
    data=df,
    x="time",
    y="stranding",
    hue="resource",
    style="resource",
    # label=f'{self._name}',
    # color=self._line_color,
    # linestyle=self._line,
)
g.set(ylim=(0, 1))
plt.savefig("packingtrace_utilization.pdf")
