#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import argparse
from re import search
from os.path import basename, getsize


COLORS = [ str(i) for i in range(20) ]
COLOR_MAP = {
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'cyan',
        5: 'violet',
        6: 'magenta',
        7: 'orange',
        8: 'brown',
        9: 'yellow',
        }
# COLORS = mcolors.CSS4_COLORS.keys()
LINES = {
    '1': '-',
    '2': '-.',
    '3': ':',
    '4': ':',
    '5': ':',
    '6': '--',
    '7': '--',
    '8': '-',
    '9': '--',
    }
# COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# Set global font size
# plt.rcParams['font.size'] = 10  # Sets the global font size to 14
# plt.rcParams['axes.labelsize'] = 10  # Sets axis label size
# plt.rcParams['xtick.labelsize'] = 8  # Sets x-tick label size
# plt.rcParams['ytick.labelsize'] = 8  # Sets y-tick label size
# plt.rcParams['legend.fontsize'] = 8  # Sets legend font size
# plt.rcParams['axes.titlesize'] = 16  # Sets title font size

class LatencyHistogram(object):
    _filepath = None
    _filename = None

    _rate = None

    _data = None
    _latencies = None

    _percentile25 = None
    _percentile50 = None
    _percentile75 = None
    _percentile99 = None

    def __init__(self, filepath):
        self._filepath = filepath
        self._filename = basename(filepath)

        self._rate = int(search(r'(\d+?)kpps', self._filename).group(1))

        self._data = np.genfromtxt(self._filepath, delimiter=',')
        self._data[:, 0] /= 1e3  # Convert to microseconds

        self._latencies = []
        for latency, count in self._data:
            self._latencies.extend([latency] * int(count))

        self._percentile25 = np.percentile(self._latencies, 25)
        self._percentile50 = np.percentile(self._latencies, 50)
        self._percentile75 = np.percentile(self._latencies, 75)
        self._percentile99 = np.percentile(self._latencies, 99)

    def filepath(self):
        return self._filepath

    def filename(self):
        return self._filename

    def rate(self):
        return self._rate

    def percentile25(self):
        return self._percentile25

    def percentile50(self):
        return self._percentile50

    def percentile75(self):
        return self._percentile75

    def percentile99(self):
        return self._percentile99


class LoadLatencyPlot(object):
    _latency_histograms = None
    _name = None
    _color = None
    _line = None
    _line_color = None

    _plot25 = None
    _plot50 = None
    _plot75 = None
    _plot99 = None

    def __init__(self, histogram_filepaths, name, color, line, line_color):
        self._latency_histograms = []
        for filepath in histogram_filepaths:
            if getsize(filepath) > 0:
                self._latency_histograms.append(LatencyHistogram(filepath))
        if len(self._latency_histograms) == 0:
            print(f"WARN: list of latency histograms empty for {histogram_filepaths}")
        self._name = name
        self._color = color
        self._line = line
        self._line_color = line_color

    def plot(self):
        hist, bin_edges = np.histogram(self._latency_histograms[0]._latencies, bins=400, density=True)
        cdf = np.cumsum(hist) * (bin_edges[1] - bin_edges[0]);
        cdf *= 100; # 1.0 -> 100%

        _x = [hist.rate() for hist in self._latency_histograms]
        _y25 = [hist.percentile25() for hist in self._latency_histograms]
        _y50 = [hist.percentile50() for hist in self._latency_histograms]
        _y75 = [hist.percentile75() for hist in self._latency_histograms]
        _y99 = [hist.percentile99() for hist in self._latency_histograms]

        order = np.argsort(_x)
        x = np.array(_x)[order]
        y25 = np.array(_y25)[order]
        y50 = np.array(_y50)[order]
        y75 = np.array(_y75)[order]
        y99 = np.array(_y99)[order]

        self._plot50 = sns.lineplot(
            x=bin_edges[1:],
            y=cdf,
            label=f'{self._name}',
            color=self._line_color,
            linestyle=self._line,
            # linewidth=1
        )


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot load latency percentile graph'
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
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help='''Path to the output plot
                             (default: load_latency.pdf)''',
                        default='load_latency.pdf'
                        )
    parser.add_argument('-c', '--compress',
                        action='store_true',
                        help='Compress the legend',
                        default=False
                        )
    for color in COLORS:
        parser.add_argument(f'--{color}',
                            type=argparse.FileType('r'),
                            nargs='+',
                            help=f'''Paths to latency histogram CSVs for
                                  {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            nargs='+',
                            help=f'''Name of {color} plot''',
                            )
    # for color in COLORS:
    #     parser.add_argument(f'--{color}-line',
    #                         type=str,
    #                         default="-",
    #                         help=f'''Line style of {color} plot''',
    #                         )
    # for color in COLORS:
    #     parser.add_argument(f'--{color}-color',
    #                         type=str,
    #                         default="blue",
    #                         help=f'''Color of {color} plot''',
    #                         )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    if not any([args.__dict__[color] for color in COLORS]):
        parser.error('At least one set of latency histogram paths must be ' +
                     'provided')

    return args


def chain(lst: list[list]) -> list:
    return [item for sublist in lst for item in sublist]


def main():
    parser = setup_parser()
    args = parse_args(parser)

    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.xlabel('Latency ($\mu s$)')
    plt.ylabel('CDF (%)')
    plt.grid()

    plots = []

    # print(vars(args))
    for color in COLORS:
        if args.__dict__[color]:
            if len(args.__dict__[f'{color}_name']) == 1:
                name = args.__dict__[f'{color}_name'][0]
                line = "-"
                line_color = "blue"
            elif len(args.__dict__[f'{color}_name']) == 3:
                name = args.__dict__[f'{color}_name'][0]
                line = args.__dict__[f'{color}_name'][1].strip("l") # allow prepending with l to avoid "-" being interpreted as a flag
                line_color = args.__dict__[f'{color}_name'][2]
            plot = LoadLatencyPlot(
                histogram_filepaths=[h.name for h in args.__dict__[color]],
                name=name,
                color=color,
                line=line,
                line_color=line_color,
            )
            plot.plot()
            plots.append(plot)

    ax.set_xscale('log' if args.logarithmic else 'linear')
    # plt.xlim(0, 1)

    legend = None

    if args.compress:
        # empty  name1 name2 ...
        # 25pctl x     x     ...
        # 50pctl x     x     ...
        # 75pctl x     x     ...
        # 99pctl x     x     ...
        dummy, = plt.plot([0], marker='None', linestyle='None',
                         label='dummy')
        legend = plt.legend(
            chain([
                [dummy, p._plot25, p._plot50, p._plot75, p._plot99]
                for p in plots
            ]),
            chain([
                [p._name, '25.pctl', '50.pctl', '75.pctl', '99.pctl']
                for p in plots
            ]),
            ncol=len(plots),
            prop={'size': 8},
            loc="lower right",
        )
    else:
        legend = plt.legend(loc="lower right", bbox_to_anchor=(1.15, 1),
                            ncol=3, title=None, frameon=False,
                            )

    ax.annotate(
        "← Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-45, -27),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    fig.tight_layout(pad=0.0)
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    main()
