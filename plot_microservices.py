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
    app: str
    color: str
    latency_median: float
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

                # microservice application
                app = str(basename(filename).split('_')[0])
                # app = "appA"

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

                # extract percentiles
                spectrum_lines = None
                for line in lines:
                    if "Detailed Percentile spectrum:" in line:
                        spectrum_lines = []
                        continue
                    if "inf" in line:
                        break # end of spectrum
                    if spectrum_lines is not None:
                        # parse
                        row = findall(r'\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                        spectrum_lines += row

                percentiles = pd.DataFrame(spectrum_lines, columns=["Value", "Percentile", "TotalCount", "1/(1-Percentile)"]).astype(float)
                # is already in ms
                percentile_50th = percentiles[percentiles.Percentile == 0.5].Value.values[0]


                # extract requests per second
                filtered = list(filter(lambda line: line.startswith("Requests/sec:"), lines))
                assert len(filtered) == 1
                line = filtered[0]
                rps = line.split(':')[1]
                rps = float(rps)
                # self.latencies[filename] = rps

                self.results[filename] = MicroserviceDataPoint(
                    name=self.name,
                    app=app,
                    color=self.color,
                    latency_median=percentile_50th,
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

    markers = []
    for hue in df['name'].unique():
        # if "Qemu" in hue:
        #     markers += [ (3, 2) ]
        if "Qemu" in hue:
            markers += [ 'x' ]
        # elif "vMux" in hue:
        #     markers += [ (5, 2) ]
        else:
            markers += [ 'o' ]

    linestyles = []
    for hue in df['name'].unique():
        if "vMux" in hue:
            linestyles += [ '-' ]
        else:
            linestyles += [ ':' ]
    # linestyles = [ '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']

    # low rpps measurements are broken
    df = df[df.offered_load_rps >= 8 ]

    # Plot using Seaborn
    grid = sns.FacetGrid(df,
            col='app',
            sharey = False,
            sharex = False,
            # gridspec_kws={"width_ratios": [11, 1]},
    )

    # wrap pointplot to set ylim
    def pointplot_with_ylim(x, y, **kwargs):
        ax = plt.gca()
        sns.pointplot(x=x, y=y, **kwargs)

        # Set different y-limits for different conditions
        if "media" in ax.get_title():
            ax.set_ylim(0, 50)
        elif "hotel" in ax.get_title():
            ax.set_ylim(0, 15)
        elif "social" in ax.get_title():
            ax.set_ylim(0, 8)

    grid.map_dataframe(pointplot_with_ylim,
    # grid.map_dataframe(sns.pointplot,
    # sns.pointplot(data=df,
                x='offered_load_rps', y='latency_median', hue='name',
                palette='colorblind',
                # kind='point',
                # capsize=.05,
                # errorbar='sd',
                errorbar=None,
                # estimator=np.median,
                markers=markers,
                linestyles=linestyles,
                )

    ax.annotate(
        "↓ Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-30, -40),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )
    grid.add_legend(
            # bbox_to_anchor=(0.5, 0.77),
            loc='right',
            ncol=1, title=None, frameon=False,
                    )
    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(0.45, 1),
    #     ncol=1,
    #     title=None,
    #     frameon=False,
    # )

    grid.figure.set_size_inches(args.width, args.height)
    # grid.set_titles("")
    plt.subplots_adjust(bottom=0.25, right=0.78)

    grid.set_xlabels(XLABEL)
    grid.set_ylabels(YLABEL)
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
