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
from plotting import HATCHES
import pickle
import os

def log(msg):
    print(msg, flush=True)

COLORS = [ str(i) for i in range(20) ]

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot packet loss graph'
    )

    parser.add_argument('-u',
                        '--utilization',
                        action='store_true',
                        help='Plot utilization instead',
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
    parser.add_argument('--cache',
                        action='store_true',
                        help='Cache crunched utilization data to /tmp'
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

def readfile(path):
    if path.endswith(".pkl"):
        df = pd.read_pickle(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)

    return df


parser = setup_parser()
args = parse_args(parser)

dfs = []
for color in COLORS:
    if args.__dict__[color]:
        arg_dfs = [ readfile(f.name) for f in args.__dict__[color] ]
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

    default_palette = sns.color_palette()
    custom_palette = ['cornflowerblue'] + list(default_palette[1:])

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
        palette=custom_palette,
        ax=ax,
    )
    # lines = g.get_lines()
    # lines[-1].set_color("lightgray") # we add the placeholder to the end of the df -> index -1

    yticks = g.get_yticks()
    yticklabels = [f"{y:.0f}k" for y in yticks]
    g.set_yticklabels(yticklabels)

    g.set_ylabel("Hardware pool\nsize")
    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=3,
        title=None,
        frameon=False,
    )

    g.annotate(
        "↓ Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-37, -32),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.85)
    plt.savefig(args.output.name)

    # stats of min/max/mean of pool_size per hue
    grouping = df.groupby("hue")
    log(grouping.describe()["pool_size"])
    fragmented = df[df["hue"] == "Fragmented"]["pool_size"].max()
    fragmented_migrate = df[df["hue"] == "Fragmented+migrate"]["pool_size"].max()
    unified = df[df["hue"] == "Unified"]["pool_size"].max()
    unified_migrate = df[df["hue"] == "Unified+migrate"]["pool_size"].max()
    normal_advantage = (float(fragmented) - float(unified)) / float(fragmented) * 100
    migrate_advantage = (float(fragmented_migrate) - float(unified_migrate)) / float(fragmented_migrate) * 100
    log(f"In the traced 14 day period, unified pools reduce the number of servers needed in the hardware pool by {migrate_advantage:.0f}% with VM migration and by {normal_advantage:.0f}% without.")
    log(f"That makes {fragmented_migrate-unified_migrate:.0f}k and {fragmented-unified:.0f}k servers less respectively.")

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

    lines = plot.get_lines()
    lines[0].set_color("lightskyblue")

    yticks = plot.get_yticks()
    yticklabels = [f"{y:.0f}k" for y in yticks]
    plot.set_yticklabels(yticklabels)

    plot.set_ylabel("Total VMs", color="cornflowerblue")
    plot.set_xlabel("Time (days)")
    plt.grid(axis='x')
    plt.xlim(0, 14)
    plt.savefig(args.output.name)
    sns.set_style("whitegrid")

    return plot


hatches = HATCHES

def barplot_add_hatches(plot_in_grid, nr_hues, offset=0):
    hatches_used = -1
    bars_hatched = 0
    for bar in plot_in_grid.patches:
        if nr_hues <= 1:
            hatches_used += 1
        else: # with multiple hues, we draw bars with the same hatch in batches
            if bars_hatched % nr_hues == 0:
                hatches_used += 1
        # if bars_hatched % 7 == 0:
        #     hatches_used += 1
        bars_hatched += 1
        if bar.get_bbox().x0 == 0 and bar.get_bbox().x1 == 0 and bar.get_bbox().y0 == 0 and bar.get_bbox().y1 == 0:
            # skip bars that are not rendered
            continue
        hatch = hatches[(offset + hatches_used) % len(hatches)]
        print(bar, hatches_used, hatch)
        bar.set_hatch(hatch)


def plot_utilization(df):

    fig = plt.figure(figsize=(args.width, args.height))

    resource_map = {
        "cores": "CPU",
        "memory": "Memory",
        "hdd": "HDD",
        "ssd": "SSD",
        "nic": "NIC",
    }

    cache_path = f"/tmp/packingtrace_utilization_{os.getuid()}.pkl"

    if args.cache and os.path.exists(cache_path):
        log(f"loading cached utilization data from {cache_path}")
        df = pickle.load(open(cache_path, "rb"))
    else:
        log("crunching utilization data")
        dfs = []
        for resource in ["cores", "memory", "hdd", "ssd", "nic"]:
            d = dict()
            d["time"] = df["time"]
            d["utilization"] = df[resource] / df["pool_size"]
            d["stranding"] = 100 * (1 - (df[resource] / df["pool_size"]))
            d["resource"] = resource_map[resource]
            d["Pool"] = df["hue"]
            dfs += [ pd.DataFrame(d) ]
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["resource"] != "HDD"]

        log("aggregating data (mean/std)")
        df_agg = df.groupby(["resource", "Pool"]).agg(
            stranding_mean=("stranding", "mean"),
            stranding_std=("stranding", "std"),
        ).reset_index()
        print(df_agg)

        # Create two synthetic points per group that preserve mean and std
        # For sample std with n=2: points at mean ± std/√2
        log("creating synthetic data points")
        rows = []
        for _, row in df_agg.iterrows():
            mean, std = row["stranding_mean"], row["stranding_std"]
            offset = std / np.sqrt(2)
            rows.append({"resource": row["resource"], "Pool": row["Pool"], "stranding": mean - offset})
            rows.append({"resource": row["resource"], "Pool": row["Pool"], "stranding": mean + offset})
        df = pd.DataFrame(rows)

        pickle.dump(df, open(cache_path, "wb"))

    default_palette = sns.color_palette("pastel")
    custom_palette = [
        default_palette[1],
        default_palette[2],
        default_palette[3],
        default_palette[4],
    ] + list(default_palette[5:])

    log("plotting utilization data")
    log("(takes maybe 20GB RAM and a few minutes)")
    g = sns.barplot(
        data=df,
        x="resource",
        y="stranding",
        hue="Pool",
        # label=f'{self._name}',
        # color=self._line_color,
        # linestyle=self._line,
        order=["CPU", "Memory", "SSD", "NIC"],
        edgecolor="dimgray",
        palette=custom_palette,
        errorbar=None,
    )
    barplot_add_hatches(g, 4)

    # Fix the legend hatches
    for i, legend_patch in enumerate(g.legend().get_patches()):
        hatch = hatches[i % len(hatches)]
        legend_patch.set_hatch(f"{hatch}{hatch}")

    g.set_ylabel("Limiting resource [%]    ")
    g.set_xlabel("Resource")

    sns.move_legend(
        g, "upper center",
        # bbox_to_anchor=(0.5, 1.45),
        ncol=2,
        title=None,
        frameon=False,
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

    g.annotate(
        "↓ Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-37, -35),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    g.set(ylim=(0, 100))
    plt.tight_layout(pad=0.1)
    plt.savefig(f"{args.output.name}")


if not args.utilization:
    g1 = add_nr_vms_plot()
    plot_poolsize(df, g1)
else:
    plot_utilization(df)


