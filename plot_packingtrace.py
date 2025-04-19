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

COLORS = [ str(i) for i in range(20) ]

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
                        default='packingtrace.pdf'
                        )
    for color in COLORS:
        parser.add_argument(f'--{color}',
                            type=argparse.FileType('r'),
                            nargs='+',
                            help=f'''Paths to MoonGen measurement logs for
                                  the {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            help=f'''Name of {color} plot''',
                            )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    if not any([args.__dict__[color] for color in COLORS]):
        parser.error('At least one set of moongen log paths must be ' +
                     'provided')

    return args


parser = setup_parser()
args = parse_args(parser)

dfs = []
for color in COLORS:
    if args.__dict__[color]:
        arg_dfs = [ pd.read_pickle(f.name) for f in args.__dict__[color] ]
        arg_df = pd.concat(arg_dfs)
        name = args.__dict__[f'{color}_name']
        arg_df["hue"] = name
        dfs += [ arg_df ]

df = pd.concat(dfs, ignore_index=True)


sns.set_theme()
sns.set_style("whitegrid")

fig = plt.figure(figsize=(args.width, args.height))

df = df[(df["time"] >= 0) & (df["time"] <= 14)]


breakpoint()

log("plot packingtrace data")
sns.lineplot(
    data=df,
    x="time",
    y="pool_size",
    hue="hue",
    style="hue",
    # label=f'{self._name}',
    # color=self._line_color,
    # linestyle=self._line,
)
plt.savefig(args.output.name)

breakpoint()


fig = plt.figure(figsize=(args.width, args.height))

log("cruching utilization data")
dfs = []
for resource in ["cores", "memory", "hdd", "ssd", "nic"]:
    d = dict()
    d["time"] = df["time"]
    d["utilization"] = df[resource] / df["pool_size"]
    d["stranding"] = 1 - (df[resource] / df["pool_size"])
    d["resource"] = resource
    d["hue"] = df["hue"]
    dfs += [ pd.DataFrame(d) ]
df = pd.concat(dfs, ignore_index=True)

breakpoint()

g = sns.barplot(
    data=df,
    x="resource",
    y="stranding",
    hue="hue",
    # label=f'{self._name}',
    # color=self._line_color,
    # linestyle=self._line,
)

# df = df[df["hue"] == "Unified"]
#
# log("plot utilization data")
# g = sns.lineplot(
#     data=df,
#     x="time",
#     y="stranding",
#     hue="resource",
#     style="resource",
#     # label=f'{self._name}',
#     # color=self._line_color,
#     # linestyle=self._line,
# )
g.set(ylim=(0, 1))
plt.savefig(f"{args.output.name}.utilization.pdf")
