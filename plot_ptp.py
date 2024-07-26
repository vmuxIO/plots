import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read data from a file and return it as a list of integers
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return data

COLORS = [ str(i) for i in range(20) ]
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
                            help=f'''Paths to PTP output files for
                                  the {color} plot''',
                            )
    for color in COLORS:
        parser.add_argument(f'--{color}-name',
                            type=str,
                            default=color,
                            help=f'''Name of {color} plot''',
                            )

    return parser


def plot():
    parser = setup_parser()
    args = parser.parse_args()

    # Directory containing the data files
    # data_dir = 'data_files'  # Change this to your directory containing the data files
    data_files = [  "/home/hendrik/vmuxIO/test/ptptest/mediation-raw.txt", 
                    "/home/hendrik/vmuxIO/test/ptptest/mediation-adjusted.txt", 
                    "/home/hendrik/vmuxIO/test/ptptest/emulation-raw.txt", 
                    "/home/hendrik/vmuxIO/test/ptptest/emulation-adjusted.txt",
                    "/home/hendrik/vmuxIO/test/ptptest/mediation-raw.txt", 
                    "/home/hendrik/vmuxIO/test/ptptest/mediation-adjusted.txt" ] # os.listdir(data_dir)

    # Initialize lists to store means and standard deviations
    means = []
    std_devs = []
    labels = []

    # Read data from each file, compute mean and standard deviation
    for i, file_path in enumerate(data_files):
        # file_path = os.path.join(data_dir, data_file)
        data = read_data(file_path)
        means.append(np.mean(data))
        std_devs.append(np.std(data))
        labels.append("File " + str(i))
    
    
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.grid()
    # plt.xlim(0, 0.83)
    ax.set_yscale('linear')

    sns.barplot(x=labels, y=means, ci=None, palette="muted")
    print(means)
    print(std_devs)
    
    for i, (mean, std) in enumerate(zip(means, std_devs)):
        plt.errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(0.45, 1),
    #     ncol=1,
    #     title=None,
    #     frameon=False,
    # )
    ax.annotate(
        "↑ Higher is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-55, -28),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)

    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    plt.tight_layout()
    plt.savefig(args.output.name)
    plt.close()


if __name__ == '__main__':
    plot()
