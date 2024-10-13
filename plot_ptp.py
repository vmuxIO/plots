import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read data from a file and return it as a list of integers
def read_data(file):
    data = [int(line.strip()) for line in file]
    return data

COLORS = [ str(i) for i in range(30) ]
YLABEL = 'Nanoseconds'
XLABEL = 'Mean and Standarddeviation'

def setup_parser():


    parser = argparse.ArgumentParser(
        description='PTP Syncing Accuracy Plot'
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
                        default='ptp.pdf'
                        )

    for color in COLORS:
        parser.add_argument(f'--{color}',
                            type=argparse.FileType('r'),
                            nargs='+',
                            help=f'''Path to PTP output files for
                                  the {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            help=f'''Name of {color} plot''',
                            )

    return parser


def filter_warmup(data):
    # We have a warmup phase where the data is not accurate, we can filter this out
    return data[20:]

def filter_packet_loss_bug(data):
    # every time we start loosing packets, we first have a huge inaccuracy of a randum but fixed value +n. Once we continue receiveing packets, we have the exactly same spike, but inverted -n.
    return [x for x in data if np.abs(x) < 20000]

def plot():
    parser = setup_parser()
    args = parser.parse_args()

    print(args.__dict__)
    names = []
    paths = []
    for color in COLORS:
        if args.__dict__[color]:
            names += [args.__dict__[f'{color}_name']]
            paths += [args.__dict__[color]]

    # Initialize lists to store means and standard deviations
    means = []
    std_devs = []
    labels = []

    width = 0.35

    # Read data from each file, compute mean and standard deviation
    for i, file_paths in enumerate(paths):
        # file_path = os.path.join(data_dir, data_file)
        data = []
        for path in file_paths:
            data_ = read_data(path)
            data_ = filter_warmup(data_)
            data_ = filter_packet_loss_bug(data_)
            data += data_

        # mean = np.abs(np.mean(data))
        mean = np.mean(np.abs(data))
        means.append(mean)
        stddev = np.std(data)
        std_devs.append(stddev)
        print(f"{i}. {names[i]}: Mean {mean} StdDev {stddev} ({YLABEL})")
        labels.append(names[i])

        x = np.arange(len(labels))

    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.grid()

    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    rects1 = ax.bar(x - width/2, means, width, label='Abs(mean)')
    rects2 = ax.bar(x + width/2, std_devs, width, label='Stdandard deviation')

    ax.set_ylabel('Nanoseconds')
    ax.set_title(args.title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')
    ax.set_yscale('log')

    def add_bar_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_bar_labels(rects1)
    add_bar_labels(rects2)

    plt.tight_layout()
    plt.savefig("output.png")

    # print some data
    def compare(i, j):
        old = np.mean([ means[n] for n in i ])
        new = np.mean([ means[n] for n in j ])
        comp_x = old/new
        comp_perc = (old-new)/new* 100
        print(f"{labels[i[0]]} vs {labels[j[0]]}: {comp_x:.1f}x, {comp_perc:.0f}%")
    compare([19, 20], [15, 16])
    compare([17], [15, 16])


    # show plot
    plt.show()


if __name__ == '__main__':
    plot()
