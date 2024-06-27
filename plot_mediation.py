#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List, Any


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
    '9_vmux-dpdk-e810_hardware': 'vmux-emu (w/ rte_flow)',
    '9_vmux-med_hardware': 'vmux-med (w/ rte_flow)',
    '9_vmux-dpdk-e810_software': 'vmux-emu',
    '9_vmux-med_software': 'vmux-med',
    '1_vfio_software': 'qemu-pt',
    '1_vmux-pt_software': 'vmux-pt',
    '1_vmux-pt_hardware': 'vmux-pt (w/ rte_flow)',
    '1_vfio_hardware': 'qemu-pt (w/ rte_flow)',
}

YLABEL = 'Rx throughput (Mpps)'
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
                        default='mediation.pdf'
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
    log_scale = (False, True) if args.logarithmic else False
    ax.set_yscale('log' if args.logarithmic else 'linear')

    dfs = []
    for color in COLORS:
        if args.__dict__[color]:
            arg_dfs = [ pd.read_csv(f.name, sep='\\s+') for f in args.__dict__[color] ]
            arg_df = pd.concat(arg_dfs)
            name = args.__dict__[f'{color}_name']
            arg_df["hue"] = name
            dfs += [ arg_df ]
            # throughput = ThroughputDatapoint(
            #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
            #     name=args.__dict__[f'{color}_name'],
            #     color=color,
            # )
            # dfs += color_dfs
    df = pd.concat(dfs)
    hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
    groups = df.groupby(hue)
    summary = df.groupby(hue)['rxMppsCalc'].describe()
    df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'fastclick', 'rate']), axis=1)
    df_hue = map_hue(df_hue, hue_map)

    # Plot using Seaborn
    bar = sns.barplot(x='num_vms', y='rxMppsCalc', hue="hue", data=pd.concat(dfs),
                palette='colorblind',
                edgecolor='dimgray',
                # kind='bar',
                # capsize=.05,  # errorbar='sd'
                # log_scale=log_scale,
                )
    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    # )
    #
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(0.45, 1),
        ncol=1,
        title=None,
        # frameon=False,
    )
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    # plt.ylim(0, 1)
    if not args.logarithmic:
        plt.ylim(bottom=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    # fig.tight_layout(rect = (0, 0, 0, 0.1))
    ax.set_position((0.1, 0.1, 0.5, 0.8))
    fig.tight_layout()
    # fig.tight_layout(rect=(0, 0, 0.3, 1))
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    main()
