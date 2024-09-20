#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from pandas import DataFrame
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List
from dataclasses import dataclass, field, asdict


COLORS = [ str(i) for i in range(20) ]
# COLORS = mcolors.CSS4_COLORS.keys()
# COLORS = [
#     'blue',
#     'cyan',
#     'green',
#     'yellow',
#     'orange',
#     'red',
#     'magenta',
# ]

hue_map = {
    '2.0_bridge_-1.0_overall': 'qemu-virtio',
    '2.0_bridge-e1000_-1.0_overall': 'qemu-e1000',
    '2.0_bridge-vhost_-1.0_overall': 'qemu-vhost',
    '2.0_vfio_-1.0_overall': 'qemu-pt',
    '2.0_vmux-dpdk-e810_-1.0_overall': 'vmux-emu-e810',
    '2.0_vmux-emu_-1.0_overall': 'vmux-emu-e1000',
    '2.0_vmux-med_-1.0_overall': 'vmux-med-e810'
}

YLABEL = 'Per-VM requests (req/s)'
XLABEL = 'Nr. of VMs'

def map_hue(df_hue, hue_map):
    return df_hue.apply(lambda row: hue_map.get(str(row), row))



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

def parse_filename(self, filename: str):
    ret = dict()
    parts = filename.split("_")
    ret['becnhmark'] = parts[0]
    ret['num_vms']


def parse_result(self, repetition: int, vm_number: int) -> DataFrame:
    def find(haystack: List[str], needle: str) -> str:
        matches = [line for line in haystack if needle in line ]
        if len(matches) != 1:
            raise Exception("Seemingly an error occured during execution")
        value = matches[0].split(" ")[-1]
        return value

    with open(self.output_path_per_vm(repetition, vm_number), 'r') as file:
        lines = file.readlines()
        data = []
        test_spec = {
            **asdict(self), # put selfs member variables and values into this dict
            "repetition": repetition,
            "vm_number": vm_number,
        }
        data += [{
            **test_spec,
            "op": "read",
            "avg_us": float(find(lines, "[READ], AverageLatency(us),")),
            "95th_us": float(find(lines, "[READ], 95thPercentileLatency(us),")),
            "99th_us": float(find(lines, "[READ], 99thPercentileLatency(us),")),
            "ops": int(find(lines, "[READ], Operations,"))
        }]
        data += [{
            **test_spec,
            "op": "update",
            "avg_us": float(find(lines, "[UPDATE], AverageLatency(us),")),
            "95th_us": float(find(lines, "[UPDATE], 95thPercentileLatency(us),")),
            "99th_us": float(find(lines, "[UPDATE], 99thPercentileLatency(us),")),
            "ops": int(find(lines, "[UPDATE], Operations,"))
        }]
        data += [{
            **test_spec, # merge test_spec dict with this dict
            "op": "overall",
            "runtime": float(find(lines, "[OVERALL], RunTime(ms),")),
            "ops_per_sec": float(find(lines, "[OVERALL], Throughput(ops/sec),"))
        }]
        return DataFrame(data=data)


def cast_column(df, column: str, caster = lambda i: int(i)):
    df[column] = df.apply(lambda row: caster(row[column]), axis=1)



def main():
    parser = setup_parser()
    args = parse_args(parser)

    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.grid()
    # plt.xlim(0, 0.83)
    ax.set_yscale('log' if args.logarithmic else 'linear')

    dfs = []
    for color in COLORS:
        if args.__dict__[color]:
            arg_dfs = [ pd.read_csv(f.name) for f in args.__dict__[color] ]
            arg_df = pd.concat(arg_dfs)
            name = args.__dict__[f'{color}_name']
            arg_df["hue"] = name
            dfs += [ arg_df ]

    df = pd.concat(dfs, ignore_index=True)
    del df['Unnamed: 0']
    df = df[df.op == 'overall'] # only the overall stat has ops_per_sec
    hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
    # groups = df.groupby(hue)
    # summary = df.groupby(hue)['rxMppsCalc'].describe()
    df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'rps', 'op']), axis=1)
    df_hue = map_hue(df_hue, hue_map)
    df['aggregate_ops_per_sec'] = df.apply(lambda row: row['num_vms'] * row['ops_per_sec'], axis=1)
    # df['aggregate_ops_per_sec'] = df.apply(lambda row: print(row), axis=1)
    # cast_column(df, 'num_vms', lambda i: int(i))
    cast_column(df, 'num_vms', int)
    markers = []
    for hue in df['hue'].unique():
        if hue == 'Qemu-pt':
            markers += [ 'o' ]
        else:
            markers += [ 'x' ]
    linestyles = []
    for hue in df['hue'].unique():
        if "vMux" in hue:
            linestyles += [ '-' ]
        else:
            linestyles += [ ':' ]
    # markers = { 'vMux-emu-e810': 'x' }
    # markers = [ 'x', '.', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x' ]
    # linestyles = [ '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']

    # Plot using Seaborn
    sns.pointplot(x='num_vms', y='ops_per_sec', hue="hue", data=df, palette='colorblind',
                # kind='point',
                capsize=.05,  # errorbar='sd'
                markers=markers,
                # linestyles=['-', '--'],
                linestyles=linestyles,
                )

    sns.move_legend(
        ax, "upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        title=None,
        frameon=False,
    )
    ax.annotate(
        "↑ Higher is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-55, -42),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )
    # plt.subplots_adjust(right=0.3)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.ylim(bottom=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    fig.tight_layout()
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    main()
