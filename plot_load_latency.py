#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from re import search
from os.path import basename, getsize


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

        self._rate = int(search(r'(\d+?)Mbps', self._filename).group(1))

        self._data = np.genfromtxt(self._filepath, delimiter=',')
        self._data[:, 0] /= 1e6

        self._latencies = []
        for latency, count in self._data:
            self._latencies.extend([latency] * int(count))

        self._percentile25 = np.percentile(self._latencies, 25)
        self._percentile50 = np.percentile(self._latencies, 50)
        self._percentile75 = np.percentile(self._latencies, 75)
        self._percentile99 = np.percentile(self._latencies, 99)
        print(self._rate, self._percentile25, self._percentile50,
              self._percentile75, self._percentile99)

    def filepath(self):
        return self._filepath

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

    def __init__(self, histogram_filepaths):
        self._latency_histograms = []
        for filepath in histogram_filepaths:
            if getsize(filepath) > 0:
                self._latency_histograms.append(LatencyHistogram(filepath))

    def plot(self):
        pass


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot load latency percentile graph'
    )

    parser.add_argument('histograms',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help='Paths to latency histogram CSVs',
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


def main():
    parser = setup_parser()
    args = parse_args(parser)

    plot = LoadLatencyPlot([h.name for h in args.histograms])
    plot.plot()


if __name__ == '__main__':
    main()
