#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def stddev(data, weights):
    return np.sqrt(np.average((data - np.average(data, weights=weights))**2,
                              weights=weights))


class DarkColorPalette(object):
    _colors = [
        'darkred',
        'saddlebrown',
        'darkorange',
        'darkgoldenrod',
        'darkolivegreen',
        'darkseagreen',
        'darkgreen',
        'darkslategray',
        'steelblue',
        'darkblue',
        'darkslateblue',
        'indigo',
        'darkmagenta',
    ]
    _index = 0

    def __init__(self, seed):
        random.Random(seed).shuffle(self._colors)

    def pull(self):
        color = self._colors[self._index]
        self._index += 1
        self._index %= len(self._colors)
        return color


class LatencyHistogram(object):
    _filepath = None
    _description = None
    _histogram_color = None
    _avxline_color = None
    _errorbar_color = None

    _data = None
    _xmax = None
    _size = None
    _weights = None
    _average = None
    _stddev = None

    def __init__(self,
                 filepath,
                 description,
                 histogram_color,
                 avxline_color,
                 errorbar_color):
        self._filepath = filepath
        self._description = description
        self._histogram_color = histogram_color
        self._avxline_color = avxline_color
        self._errorbar_color = errorbar_color

        self._data = np.genfromtxt(self._filepath, delimiter=',')
        self._data[:, 0] /= 1e6
        self._xmax = max(self._data[:, 0])
        self._size = int(sum(self._data[:, 1]))
        self._weights = np.ones_like(self._data[:, 1]) / len(self._data[:, 0])
        self._average = np.average(self._data[:, 0], weights=self._weights)
        self._stddev = stddev(self._data[:, 0], self._weights)

    def filepath(self):
        return self._filepath

    def description(self):
        return self._description

    def histogram_color(self):
        return self._histogram_color

    def avxline_color(self):
        return self._avxline_color

    def errorbar_color(self):
        return self._errorbar_color

    def data(self):
        return self._data

    def xmax(self):
        return self._xmax

    def size(self):
        return self._size

    def weights(self):
        return self._weights

    def average(self):
        return self._average

    def stddev(self):
        return self._stddev


class LatencyHistogramPlot(object):
    _latency_histograms = None
    _color_palette = DarkColorPalette(1)

    _xmin = 0.
    _xmax = None
    _histrange = None

    _bins = 300

    _fig = None
    _ax = None

    def __init__(self, filepaths, descriptions):
        assert len(filepaths) == len(descriptions)

        self._latency_histograms = []
        for filepath, description in zip(filepaths, descriptions):
            histogram_color = self._color_palette.pull()
            avxline_color = lighten_color(histogram_color, amount=0.5)
            errorbar_color = lighten_color(histogram_color, amount=0.75)
            self._latency_histograms.append(
                LatencyHistogram(
                    filepath,
                    description,
                    histogram_color,
                    avxline_color,
                    errorbar_color
                )
            )
        self._xmax = 1.1 * max(
            [hist.xmax() for hist in self._latency_histograms]
        )
        self._histrange = (self._xmin, self._xmax)

    def plot(self, output_filepath):
        self._fig = plt.figure(figsize=(12, 6))
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_axisbelow(True)
        plt.title('Latency Histogram for different Network Interfaces')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')

        self._ax.set_xticks(np.arange(self._xmin, self._xmax, 1.0))
        self._ax.set_xticks(np.arange(self._xmin, self._xmax, 0.25),
                            minor=True)
        self._ax.set_yticks(np.arange(0., 1.1, 0.1))
        self._ax.set_yticks(np.arange(0., 1.1, 0.025), minor=True)
        plt.grid(which='major', alpha=0.5, linestyle='dotted', linewidth=0.5)
        plt.grid(which='minor', alpha=0.2, linestyle='dotted', linewidth=0.5)

        for hist in self._latency_histograms:
            plt.hist(
                hist.data()[:,0],
                range=self._histrange,
                weights=hist.weights(),
                bins=self._bins,
                linewidth=0.5,
                edgecolor='black',
                facecolor=hist.histogram_color(),
                label=hist.description(),
            )

        for hist in self._latency_histograms:
            plt.axvline(
                x=hist.average(),
                color=hist.avxline_color(),
                linewidth=1.0,
                label=(f'Average for {hist.description()}: ' +
                       f'{hist.average():.2f} ms'),
            )

        for hist in self._latency_histograms:
            plt.errorbar(
                hist.average(),
                0.5,
                xerr=hist.stddev(),
                fmt='o',
                color=hist.errorbar_color(),
                markersize=0,
                capsize=5,
                capthick=1,
                label=(f'Std. Dev. for {hist.description()}: ' +
                       f'{hist.stddev():.2f} ms'),
            )

        legend = plt.legend()
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        plt.savefig(output_filepath)
        plt.close()


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot latency histograms for different network interfaces'
    )

    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help='''Path to the output plot
                             (default: latency_histogram.pdf)''',
                        default='latency_histograms.pdf'
                        )
    parser.add_argument('-f', '--files',
                        action='append',
                        type=argparse.FileType('r'),
                        help='Paths to latency histogram CSVs',
                        required=True,
                        )
    parser.add_argument('-d', '--descriptions',
                        action='append',
                        type=str,
                        help='Descriptions for latency histograms',
                        required=True,
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    assert len(args.files) == len(args.descriptions)

    return args


def main():
    parser = setup_parser()
    args = parse_args(parser)

    plot = LatencyHistogramPlot(args.files, args.descriptions)
    plot.plot(args.output.name)


if __name__ == '__main__':
    main()
