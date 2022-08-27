#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
from re import search, findall, MULTILINE
from os.path import basename, getsize


class MoonGenLog(object):
    _filepath = None
    _filename = None

    _rate = None

    _tx_avg = None
    _tx_stddev = None
    _rx_avg = None
    _rx_stddev = None

    _packet_loss_avg = None
    _packet_loss_stddev = None

    def __init__(self, filepath):
        self._filepath = filepath
        self._filename = basename(filepath)

        self._rate = int(search(r'(\d+?)Mbps', self._filename).group(1))

        log = ''
        with open(self._filepath, 'r') as f:
            log = f.read()

        rgx_summary_line = r'^.* bytes \(incl. CRC\)$'
        rx_line, tx_line = findall(rgx_summary_line, log, flags=MULTILINE)
        assert 'TX' in tx_line and 'RX' in rx_line, \
            'Could not find TX and RX lines in log'

        rgx_rate = r'(\d+?) \(StdDev (\d+?)\) Mbit/s'
        tx_search = search(rgx_rate, tx_line)
        rx_search = search(rgx_rate, rx_line)

        self._tx_avg = int(tx_search.group(1))
        self._tx_stddev = int(tx_search.group(2))
        self._rx_avg = int(rx_search.group(1))
        self._rx_stddev = int(rx_search.group(2))

        assert self._tx_stddev == 0, \
            f'TX StdDev is not 0, take measurement {self._filepath} again'

        self._packet_loss_avg = 100 * (self._tx_avg - self._rx_avg) \
            / self._tx_avg
        self._packet_loss_stddev = 100 * self._rx_stddev / self._tx_avg

    def filepath(self):
        return self._filepath

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
            if getsize(filepath) > 0:
                self._moongen_logs.append(MoonGenLog(filepath))

    def plot(self, color='blue'):
        _x = [log.rate() for log in self._moongen_logs]
        _y = [log.packet_loss_avg() for log in self._moongen_logs]
        _yerr = [log.packet_loss_stddev() for log in self._moongen_logs]

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
    plt.title('Packet loss for different loads')
    plt.xlabel('Load (Mbps)')
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
