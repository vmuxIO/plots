#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from re import search, findall, MULTILINE
from os.path import basename, getsize


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

        rgx_rate = r'(\d+?) \(StdDev (\d+?)\) Mbit/s'
        tx_search = search(rgx_rate, tx_line)
        rx_search = search(rgx_rate, rx_line)

        self._tx_avg = int(tx_search.group(1))
        self._tx_stddev = int(tx_search.group(2))
        self._rx_avg = int(rx_search.group(1))
        self._rx_stddev = int(rx_search.group(2))

        if self._tx_stddev > 0.2 * self._tx_avg:
            print(f'Invalid log file: {self._filepath}, ' +
                  'TX rate has too large standard deviation, skipping...')
            self._valid = False
            return

        self._packet_loss_avg = 100 * (self._tx_avg - self._rx_avg) \
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


class PacketLossPlot(object):
    _moongen_logs = None

    def __init__(self, moongen_log_filepaths):
        self._moongen_logs = []
        for filepath in moongen_log_filepaths:
            moongen_log = MoonGenLog(filepath)
            if not moongen_log.valid():
                continue
            self._moongen_logs.append(moongen_log)

    def plot(self, color='blue'):
        data = {}
        for moongen_log in self._moongen_logs:
            rate = moongen_log.rate()
            if rate not in data:
                data[rate] = []
            data[rate].append(moongen_log)

        _x = []
        _y = []
        _yerr = []

        for rate, logs in data.items():
            _x.append(rate)
            _y.append(np.mean([log.packet_loss_avg() for log in logs]))
            _yerr.append(np.mean([log.packet_loss_stddev() for log in logs]))

        order = np.argsort(_x)
        x = np.array(_x)[order]
        y = np.array(_y)[order]
        yerr = np.array(_yerr)[order]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            ecolor=color,
            linewidth=1,
            linestyle='--',
            marker='o',
            elinewidth=1,
            capsize=5,
        )


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Plot packet loss graph'
    )

    parser.add_argument('-t',
                        '--title',
                        type=str,
                        help='Title of the plot',
                        )
    parser.add_argument('logs',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help='Paths to MoonGen measurement logs',
                        )
    parser.add_argument('-o', '--output',
                        type=argparse.FileType('w+'),
                        help='''Path to the output plot
                             (default: packet_loss.pdf)''',
                        default='packet_loss.pdf'
                        )
    parser.add_argument('-c', '--compare',
                        type=argparse.FileType('r'),
                        nargs='+',
                        help='Paths to MoonGen measurement logs to compare to',
                        )

    return parser


def parse_args(parser):
    args = parser.parse_args()

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
    plt.ylabel('Packet Loss (%)')
    plt.grid()
    plt.ylim(-5, 105)

    plot = PacketLossPlot([h.name for h in args.logs])
    plot.plot(color='blue')

    if args.compare:
        plot2 = PacketLossPlot([h.name for h in args.compare])
        plot2.plot(color='orange')

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    plt.savefig(args.output.name)
    plt.close()

if __name__ == '__main__':
    main()
