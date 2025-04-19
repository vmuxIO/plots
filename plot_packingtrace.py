import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass
import argparse
from math import nan as NaN

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

df = df[(df["time"] >= 0) & (df["time"] <= 14)]

sns.set_theme()
sns.set_style("whitegrid")


def plot_poolsize(df, g_nr_vms_plot):
    ax = g_nr_vms_plot.twinx()

    placeholder = dict()
    for colname in df.columns:
        placeholder[colname] = [ NaN ]
    placeholder["hue"] = [ "Total VMs" ]
    df = pd.concat([pd.DataFrame(placeholder), df], ignore_index=True)


    log("plot packingtrace data")
    df["pool_size"] = df["pool_size"] / 1000
    g = sns.lineplot(
        data=df,
        x="time",
        y="pool_size",
        hue="hue",
        style="hue",
        # label=f'{self._name}',
        # color=self._line_color,
        # linestyle=self._line,
        ax=ax,
    )
    g.set_ylabel("Hardware pool\nsize (thousand)")
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=3,
        title=None,
        frameon=False,
    )
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.9)
    plt.savefig(args.output.name)

    return g


def add_nr_vms_plot():
    fig = plt.figure(figsize=(args.width, args.height))

    # Connect to the database
    conn = sqlite3.connect('../packing_trace_zone_a_v1.sqlite')

    # Load VM requests

    log("Loading trace data")
    vm_df = pd.read_sql_query("SELECT vmId, vmTypeId, starttime, endtime FROM vm", conn)

    # Generate day range (0 to 14 for the 14-day period)
    days = range(150)  # 0-14 inclusive

    # Initialize results storage
    results_by_type = []
    active_counts = []
    times = []


    log("Parsing")
    # For each day, count active VMs by type
    for day in tqdm(days):
        day = float(day)/10.0
        # A VM is active if day >= starttime AND (endtime is NULL OR day <= endtime)
        active_mask = (vm_df['starttime'] <= day) & ((vm_df['endtime'].isnull()) | (vm_df['endtime'] >= day))
        active_vms = vm_df[active_mask]

        # Group by VM type and count
        type_counts = active_vms.groupby('vmTypeId').size().reset_index()
        type_counts.columns = ['vmTypeId', 'active_count']
        type_counts['day'] = day

        # only take the most common types
        n = 10
        type_counts = type_counts.sort_values("active_count", ascending=False).head(n)

        results_by_type.append(type_counts)
        active_counts += [ len(active_vms) ]
        times += [ day ]

    # Combine all results
    # df = pd.concat(results_by_type)
    df = pd.DataFrame({ 'day': times, 'active_count': active_counts })


    log("plotting VM count data")
    sns.set_style("white")
    df["active_count"] = df["active_count"] / 1_000
    plot = sns.lineplot(
        data=df,
        # x=bin_edges[1:],
        # y=cdf,
        x = "day",
        y = "active_count",
        # hue = "vmTypeId",
        # style = "vmTypeId",
        # label=f'{self._name}',
        # color=self._line_color,
        # linestyle=self._line,
        # linewidth=1,
        markers=True,
        # errorbar='ci',
        # markers=[ 'X' ],
        # markeredgecolor='black',
        # markersize=60,
        # markeredgewidth=1,
    )
    plot.set_ylabel("Total VMs (thousand)")
    plot.set_xlabel("Time (days)")
    plt.grid(axis='x')
    plt.xlim(0, 14)
    plt.savefig(args.output.name)
    sns.set_style("whitegrid")

    return plot


def plot_utilization(df):
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

    log("plotting utilization data")
    log("(takes >30GB RAM and a few minutes)")
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



g1 = add_nr_vms_plot()
plot_poolsize(df, g1)
# plot_utilization(df)


