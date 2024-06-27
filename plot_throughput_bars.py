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


COLORS = mcolors.CSS4_COLORS.keys()
# COLORS = [
#     'blue',
#     'cyan',
#     'green',
#     'yellow',
#     'orange',
#     'red',
#     'magenta',
# ]
TARGET_PACKET_LOSS = 1 # target packet loss in percent

# # Example for find_x_intersections
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([1, 3, 2, 2.5, 0.5, 3])
# y_target = 2
#
# print(find_x_intersections(x, y, y_target)) # [0.5, 3.25, 4.6]
def find_x_intersections(x, y, y_target):
    # Identify segments where line crosses y_target
    below = y < y_target
    above = y > y_target
    cross = np.where(below[:-1] & above[1:] | above[:-1] & below[1:])[0]

    x_intersections = []

    # For each crossing segment, interpolate to find x value of intersection
    for idx in cross:
        dy = y[idx+1] - y[idx]
        if dy != 0:  # Avoid division by zero
            dx = x[idx+1] - x[idx]
            factor = (y_target - y[idx]) / dy
            x_intersection = x[idx] + factor * dx
            x_intersections.append(x_intersection)
    
    return x_intersections

def find_x_intersection(x, y, y_target):
    x_intersections = find_x_intersections(np.array(x), np.array(y), y_target)
    if len(x_intersections) != 1:
        print(f'WARN: Invalid number of intersections at x={x_intersections}')
    if len(x_intersections) == 0:
        # assuming that the function is monotonically increasing:
        if y_target < y[0]:
            return x[0]
        else:
            return x[-1]
    return x_intersections[0]

def explode(mean: float, stddev: float) -> List[float]:
    """
    return a list of values that have @mean with @stddev
    """
    return [mean + stddev, mean - stddev]

def stddev_to_series(df: pd.DataFrame, mean: str, stddev: str) -> pd.DataFrame:
    all_rows = []
    for _index, row in df.iterrows():
        samples = explode(row[mean], row[stddev])
        temp_df = pd.DataFrame([row] * len(samples))
        temp_df[mean] = samples
        all_rows.append(temp_df)
        
    ret = pd.concat(all_rows, ignore_index=True)
    del ret[stddev]
    return ret

class MoonGenLog(object):
    _filepath = None
    _filename = None

    _valid = None

    _rate = None

    _tx_avg = None
    _tx_stddev = None
    _rx_avg = None
    _rx_stddev = None

    _packet_loss_avg = None
    _packet_loss_stddev = None

    def __init__(self, filepath):
        self._filepath = filepath

        if getsize(self._filepath) <= 0:
            print(f'Invalid log file: {self._filepath}, ' +
                  'file is empty, skipping...')
            self._valid = False
            return

        self._filename = basename(filepath)

        self._rate = int(search(r'(\d+?)kpps', self._filename).group(1))

        log = ''
        with open(self._filepath, 'r') as f:
            log = f.read()

        rgx_summary_line = r'^.* bytes \(incl. CRC\)$'
        rx_line, tx_line = findall(rgx_summary_line, log, flags=MULTILINE)
        if not ('TX' in tx_line and 'RX' in rx_line):
            print(f'Invalid log file: {self._filepath}, ' +
                  'TX or RX summary not found, skipping...')
            self._valid = False
            return

        # bitrate
        # rgx_rate = r'(\d+?) \(StdDev (\d+?)\) Mbit/s'
        # self._tx_avg = int(tx_search.group(1))
        # ...

        # packet rate
        rgx_rate = r'(\d+?)\.(\d+?) \(StdDev (\d+?)\.(\d+?)\) Mpps,'
        tx_search = search(rgx_rate, tx_line)
        rx_search = search(rgx_rate, rx_line)

        # this is kpps now instead of bitrate
        self._tx_avg =    float(f'{tx_search.group(1)}.{tx_search.group(2)}') * 1000
        self._tx_stddev = float(f'{tx_search.group(3)}.{tx_search.group(4)}') * 1000
        self._rx_avg =    float(f'{rx_search.group(1)}.{rx_search.group(2)}') * 1000
        self._rx_stddev = float(f'{rx_search.group(3)}.{rx_search.group(4)}') * 1000

        if self._tx_stddev > 0.2 * self._tx_avg:
            print(f'Invalid log file: {self._filepath}, ' +
                  'TX rate has too large standard deviation, skipping...')
            self._valid = False
            return

        if self._tx_avg == 0:
            print(f"WARN: tx avg is 0 for {filepath}")
        self._packet_loss_avg = 100 * max(self._tx_avg - self._rx_avg, 0) \
            / self._tx_avg
        self._packet_loss_stddev = 100 * self._rx_stddev / self._tx_avg

        self._valid = True

    def filepath(self):
        return self._filepath

    def filename(self):
        return self._filename

    def valid(self):
        return self._valid

    def rate(self):
        return self._rate

    def tx_avg(self):
        return self._tx_avg

    def tx_stddev(self):
        return self._tx_stddev

    def rx_avg(self):
        return self._rx_avg

    def rx_stddev(self):
        return self._rx_stddev

    def packet_loss_avg(self):
        return self._packet_loss_avg

    def packet_loss_stddev(self):
        return self._packet_loss_stddev


class ThroughputDatapoint(object):
    _moongen_logs = None
    _name = None
    _color = None
    y_max = None
    y_max_err = None
    y_tpl = None
    y_tpl_err = None
    df = None

    def __init__(self, moongen_log_filepaths, name, color):
        self._moongen_logs = []
        for filepath in moongen_log_filepaths:
            moongen_log = MoonGenLog(filepath)
            if not moongen_log.valid():
                continue
            self._moongen_logs.append(moongen_log)
        self._name = name
        self._color = color

        self.find_throughputs()

        self.prepare_df()

    def find_throughputs(self):
        data = {}
        for moongen_log in self._moongen_logs:
            rate = moongen_log.rate()
            if rate not in data:
                data[rate] = []
            data[rate].append(moongen_log)

        _x = []
        _y_loss = []
        _y_loss_err = []
        _y = []
        _yerr = []

        for rate, logs in data.items():
            _x.append(rate)
            _y_loss.append(np.mean([log.packet_loss_avg() for log in logs]))
            _y_loss_err.append(np.mean([log.packet_loss_stddev() for log in logs]))
            _y.append(np.mean([log.rx_avg() for log in logs]))
            _yerr.append(np.mean([log.rx_stddev() for log in logs]))

        order = np.argsort(_x)
        x = np.array(_x)[order]
        y = np.array(_y)[order]
        yerr = np.array(_yerr)[order]
        y_loss = np.array(_y_loss)[order]
        y_loss_err = np.array(_y_loss_err)[order]

        # find max throuhgput
        position = np.argmax(y)
        self.y_max = y[position]
        self.y_max_err = yerr[position]

        # find throughput at n% packet loss
        x_tpl = find_x_intersection(x, y_loss, TARGET_PACKET_LOSS)
        self.y_tpl = np.interp(x_tpl, x, y) # throughput at target packet loss
        self.y_tpl_err = np.interp(x_tpl, x, yerr) # throughput at target packet loss

        # # plot packet loss to offered packet rate
        # plt.errorbar(
        #     x,
        #     y,
        #     yerr=yerr,
        #     label=self._name,
        #     color=self._color,
        #     ecolor=self._color,
        #     linewidth=1,
        #     linestyle='--',
        #     marker='o',
        #     elinewidth=1,
        #     capsize=5,
        # )

    def prepare_df(self):
        # Sample data
        categories = [str(TARGET_PACKET_LOSS), 'any']
        values = [self.y_tpl, self.y_max]
        errors = [self.y_tpl_err, self.y_max_err]
        
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame({'Category': categories, 'Values': values, 'Stderr': errors, 'Group': self._name})

        # prepare stderr for seaborn
        self.df = stddev_to_series(df, "Values", "Stderr")



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
                        default='packet_loss.pdf'
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
    # plt.ylim(-5, 105)
    ax.set_yscale('log' if args.logarithmic else 'linear')

    dfs = []
    for color in COLORS:
        if args.__dict__[color]:
            throughput = ThroughputDatapoint(
                moongen_log_filepaths=[h.name for h in args.__dict__[color]],
                name=args.__dict__[f'{color}_name'],
                color=color,
            )
            dfs += [throughput.df]
        
    df = pd.concat(dfs)
            
    # Plot using Seaborn
    sns.barplot(x='Category', y='Values', hue='Group', data=df, palette='colorblind', edgecolor='black')
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    )

    plt.xlabel('Packet Loss (%)')
    plt.ylabel('Throughput (kpps)')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')
    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    fig.tight_layout()
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    main()
