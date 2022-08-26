#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse


class LoadLatencyPlot(object):
    def __init__(self):
        pass

    def plot(self):
        pass


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot load latency percentile graph'
    )

    return parser


def parse_args(parser):
    args = parser.parse_args()

    return args


def main():
    parser = setup_parser()
    args = parse_args(parser)

    plot = LoadLatencyPlot()
    plot.plot()


if __name__ == '__main__':
    main()
