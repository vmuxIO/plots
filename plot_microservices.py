#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List
from dataclasses import dataclass


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
    '9_vmux-dpdk-e810_hardware': 'vmux-emu (hardware)',
    '9_vmux-med_hardware': 'vmux-med (hardware)',
    '9_vmux-dpdk-e810_software': 'vmux-emu (software)',
    '9_vmux-med_software': 'vmux-med (software)',
}

YLABEL = 'Latency (ms)'
XLABEL = 'Offered load (req/s)'

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
                        default='microservices.pdf'
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


@dataclass
class MicroserviceDataPoint:
    name: str
    color: str
    latency_mean: float
    latency_stddev: float
    rps: float
    offered_load_rps: int

class MicroserviceTest:
    def __init__(self, log_filepaths: List[str], name: str, color: str):
        self.log_filepaths = log_filepaths
        self.name = name
        self.color = color
        self.results = dict()
        # self.latencies = dict() # (mean in ms, stddev)
        # self.rps = dict() # requests per second

        for filename in log_filepaths:
            with open(filename, 'r') as f:
                lines = f.readlines()

                # extract offered_load_rps
                offered_load_rps = filename.split('_')[-2]
                offered_load_rps = offered_load_rps.split('rps')[0]
                offered_load_rps = int(offered_load_rps)

                # extract latencies
                filtered = list(filter(lambda line: line.startswith("#[Mean"), lines))
                if len(filtered) != 1:
                    print(f'Warning: Skipping {filename}. It doesnt contain valid measurement results.')
                    continue
                line = filtered[0]
                inner = line.strip("#[]\n \t")
                key_values = inner.split(',')
                mean_value = key_values[0].split("=")[1]
                stddev_value = key_values[1].split("=")[1]
                mean_value = float(mean_value)
                stddev_value = float(stddev_value)
                # self.latencies[filename] = (mean_value, stddev_value)

                # extract requests per second
                filtered = list(filter(lambda line: line.startswith("Requests/sec:"), lines))
                assert len(filtered) == 1
                line = filtered[0]
                rps = line.split(':')[1]
                rps = float(rps)
                # self.latencies[filename] = rps

                self.results[filename] = MicroserviceDataPoint(
                    name=self.name,
                    color=self.color,
                    latency_mean=mean_value,
                    latency_stddev=stddev_value,
                    rps=rps,
                    offered_load_rps=offered_load_rps,
                )

    def toDataFrame(self):
        return pd.DataFrame([vars(v) for v in self.results.values()])

    def __str__(self):
        return f'{self.name} ({self.color})'

    def __repr__(self):
        return self.__str__()


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

    data_lines = []
    for color in COLORS:
        if args.__dict__[color]:
            data_lines += [MicroserviceTest(
                log_filepaths=[f.name for f in args.__dict__[color]],
                name=args.__dict__[f'{color}_name'],
                color=color,
            )]
            # dfs += [ pd.read_csv(f.name, sep='\\s+') for f in args.__dict__[color] ]
            # throughput = ThroughputDatapoint(
            #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
            #     name=args.__dict__[f'{color}_name'],
            #     color=color,
            # )
            # dfs += color_dfs

    dfs = [d.toDataFrame() for d in data_lines]
    df = pd.concat(dfs)
    # hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
    # groups = df.groupby(hue)
    # summary = df.groupby(hue)['rxMppsCalc'].describe()
    # df_hue = df.apply(lambda row: '_'.join(str(row.color), str(row.name)), axis=1)
    # df_hue = map_hue(df_hue, hue_map)

    # Plot using Seaborn
    sns.catplot(x='offered_load_rps', y='latency_mean', hue=df['name'], data=df,
                palette='colorblind',
                kind='point',
                capsize=.05,  # errorbar='sd'
                )
    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    # )
    #
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.ylim(0, 10)
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
