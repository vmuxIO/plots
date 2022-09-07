#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from re import search
from os.path import basename, getsize


COLORS = [
    'blue',
    'cyan',
    'green',
    'yellow',
    'orange',
    'red',
    'magenta',
]


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
        self._data[:, 0] /= 1e6

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

    def __init__(self, histogram_filepaths, name, color):
        self._latency_histograms = []
        for filepath in histogram_filepaths:
            if getsize(filepath) > 0:
                self._latency_histograms.append(LatencyHistogram(filepath))
        self._name = name
        self._color = color

    def plot(self):
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

        plt.plot(
            x,
            y25,
            label=f'{self._name} 25th percentile',
            linestyle=':',
            color=self._color,
            linewidth=1,
        )
        plt.plot(
            x,
            y50,
            label=f'{self._name} 50th percentile',
            color=self._color,
            linestyle='-',
            linewidth=1,
            marker='x',
        )
        plt.plot(
            x,
            y75,
            label=f'{self._name} 75th percentile',
            color=self._color,
            linestyle='-.',
            linewidth=1,
        )
        plt.plot(
            x,
            y99,
            label=f'{self._name} 99th percentile',
            linestyle='--',
            color=self._color,
            linewidth=1,
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
                            help=f'''Name of {color} plot''',
                            )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    if not any([args.__dict__[color] for color in COLORS]):
        parser.error('At least one set of latency histogram paths must be ' +
                     'provided')

    return args


def main():
    parser = setup_parser()
    args = parse_args(parser)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.xlabel('Load (kpps)')
    plt.ylabel('Latency (ms)')
    plt.grid()

    for color in COLORS:
        if args.__dict__[color]:
            plot = LoadLatencyPlot(
                histogram_filepaths=[h.name for h in args.__dict__[color]],
                name=args.__dict__[f'{color}_name'],
                color=color
            )
            plot.plot()

    ax.set_yscale('log' if args.logarithmic else 'linear')

    legend = plt.legend()
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    main()
