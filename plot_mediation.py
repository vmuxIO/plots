#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import seaborn as sns
import pandas as pd
from re import search, findall, MULTILINE
from os.path import basename, getsize
from typing import List, Any
from plotting import HATCHES as hatches


COLORS = [ str(i) for i in range(20) ]
# COLORS = mcolors.CSS4_COLORS.keys()
# COLORS = [
#     'blue',
#     'cyan',
#     'green',
#     'yellow',
#     'orange',
#     'red',
#     'magenta',
# ]

hue_map = {
    '9_vmux-dpdk-e810_hardware': 'vmux-emu (w/ rte_flow)',
    '9_vmux-med_hardware': 'vmux-med (w/ rte_flow)',
    '9_vmux-dpdk-e810_software': 'vmux-emu',
    '9_vmux-med_software': 'vmux-med',
    '1_vfio_software': 'qemu-pt',
    '1_vmux-pt_software': 'vmux-pt',
    '1_vmux-pt_hardware': 'vmux-pt (w/ rte_flow)',
    '1_vfio_hardware': 'qemu-pt (w/ rte_flow)',
}

YLABEL = 'Rx throughput (Mpps)'
XLABEL = 'Nr. of VMs'

def map_hue(df_hue, hue_map):
    return df_hue.apply(lambda row: hue_map.get(str(row), row))



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
                        default='mediation.pdf'
                        )
    parser.add_argument('-l', '--logarithmic',
                        action='store_true',
                        help='Plot logarithmic latency axis',
                        )
    parser.add_argument('-s', '--slides',
                        action='store_true',
                        help='Use other setting to plot for presentation slides',
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

# hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O']
hatches_used = 0

# Define a custom function to add hatches to the bar plots
def barplot_with_hatches(*args, **kwargs):
    global hatches_used
    sns.barplot(*args, **kwargs)
    for i, bar in enumerate(plt.gca().patches):
        hatch = hatches[hatches_used % len(hatches)]
        print(hatch)
        bar.set_hatch(hatch)
        hatches_used += 1


def main():
    parser = setup_parser()
    args = parse_args(parser)

    # fig = plt.figure(figsize=(args.width, args.height))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_axisbelow(True)
    if args.title:
        plt.title(args.title)
    plt.grid()
    # plt.xlim(0, 0.83)
    log_scale = (False, True) if args.logarithmic else False
    # ax.set_yscale('log' if args.logarithmic else 'linear')

    dfs = []
    for color in COLORS:
        if args.__dict__[color]:
            arg_dfs = [ pd.read_csv(f.name, sep='\\s+') for f in args.__dict__[color] ]
            arg_df = pd.concat(arg_dfs)
            name = args.__dict__[f'{color}_name']
            arg_df["hue"] = name
            dfs += [ arg_df ]
            # throughput = ThroughputDatapoint(
            #     moongen_log_filepaths=[f.name for f in args.__dict__[color]],
            #     name=args.__dict__[f'{color}_name'],
            #     color=color,
            # )
            # dfs += color_dfs
    df = pd.concat(dfs)
    hue = ['repetitions', 'num_vms', 'interface', 'fastclick']
    groups = df.groupby(hue)
    summary = df.groupby(hue)['rxMppsCalc'].describe()
    df_hue = df.apply(lambda row: '_'.join(str(row[col]) for col in ['repetitions', 'interface', 'fastclick', 'rate']), axis=1)
    df_hue = map_hue(df_hue, hue_map)
    df['is_passthrough'] = df.apply(lambda row: True if "vmux-pt" in row['interface'] or "vfio" in row['interface'] else False, axis=1)

    # map colors to hues
    colors = sns.color_palette("pastel", len(df['hue'].unique())-1) + [ mcolors.to_rgb('sandybrown') ]
    palette = dict(zip(df['hue'].unique(), colors))

    # Plot using Seaborn
    grid = sns.FacetGrid(df,
            col='is_passthrough',
            sharey = False,
            sharex = False,
            gridspec_kws={"width_ratios": [11, 1]},
    )
    grid.map_dataframe(sns.barplot,
    # grid.map_dataframe(barplot_with_hatches,
                       x='num_vms',
                       y='rxMppsCalc',
                       hue="hue",
                       palette=palette,
                       edgecolor="dimgray",
                       )

    grid.add_legend(
            # bbox_to_anchor=(0.5, 0.77),
            loc='right',
            ncol=1, title=None, frameon=False,
                    )

    # Fix the legend hatches
    for i, legend_patch in enumerate(grid._legend.get_patches()):
        hatch = hatches[i % len(hatches)]
        legend_patch.set_hatch(f"{hatch}{hatch}")

    # add hatches to bars
    for (i, j, k), data in grid.facet_data():
        print(i, j, k)
        def barplot_add_hatches(plot_in_grid, nr_hues, offset=0):
            hatches_used = -1
            bars_hatched = 0
            for bar in plot_in_grid.patches:
                if nr_hues <= 1:
                    hatches_used += 1
                else: # with multiple hues, we draw bars with the same hatch in batches
                    if bars_hatched % nr_hues == 0:
                        hatches_used += 1
                # if bars_hatched % 7 == 0:
                #     hatches_used += 1
                bars_hatched += 1
                if bar.get_bbox().x0 == 0 and bar.get_bbox().x1 == 0 and bar.get_bbox().y0 == 0 and bar.get_bbox().y1 == 0:
                    # skip bars that are not rendered
                    continue
                hatch = hatches[(offset + hatches_used) % len(hatches)]
                print(bar, hatches_used, hatch)
                bar.set_hatch(hatch)

        if (i, j, k) == (0, 0, 0):
            barplot_add_hatches(grid.facet_axis(i, j), 7)
        elif (i, j, k) == (0, 1, 0):
            barplot_add_hatches(grid.facet_axis(i, j), 1, offset=(7 if not args.slides else 4))

    def grid_set_titles(grid, titles):
        for ax, title in zip(grid.axes.flat, titles):
            ax.set_title(title)

    grid_set_titles(grid, ["Emulation and Mediation", "Passthrough"])

    grid.figure.set_size_inches(args.width, args.height)
    # grid.set_titles("foobar")
    plt.subplots_adjust(left=0.06)
    # bar = sns.barplot(x='num_vms', y='rxMppsCalc', hue="hue", data=pd.concat(dfs),
    #             palette='colorblind',
    #             edgecolor='dimgray',
    #             # kind='bar',
    #             # capsize=.05,  # errorbar='sd'
    #             # log_scale=log_scale,
    #             ax=ax,
    #             )
    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    # )
    #
    # sns.move_legend(
    #     grid, "lower center",
    #     bbox_to_anchor=(0.45, 1),
    #     ncol=1,
    #     title=None,
    #     # frameon=False,
    # )
    grid.set_xlabels(XLABEL)
    grid.set_ylabels(YLABEL)

    grid.facet_axis(0, 0).annotate(
        "↓ Lower is better", # or ↓ ← ↑ →
        xycoords="axes points",
        # xy=(0, 0),
        xy=(0, 0),
        xytext=(-37, -28),
        # fontsize=FONT_SIZE,
        color="navy",
        weight="bold",
    )

    # plt.ylim(0, 1)
    if not args.logarithmic:
        plt.ylim(bottom=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.0f')

    # # iterate through each container, hatch, and legend handle
    # for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles[::-1]):
    #     # update the hatching in the legend handle
    #     handle.set_hatch(hatch)
    #     # iterate through each rectangle in the container
    #     for rectangle in container:
    #         # set the rectangle hatch
    #         rectangle.set_hatch(hatch)

    # # Loop over the bars
    # for i,thisbar in enumerate(bar.patches):
    #     # Set a different hatch for each bar
    #     thisbar.set_hatch(hatches[i % len(hatches)])

    # legend = plt.legend()
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(0.8)
    # fig.tight_layout(rect = (0, 0, 0, 0.1))
    # ax.set_position((0.1, 0.1, 0.5, 0.8))
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(right=0.78)
    # fig.tight_layout(rect=(0, 0, 0.3, 1))
    plt.savefig(args.output.name)
    plt.close()

    # calulate numbers to include in text
    print("")

    pt_rteflow = df[(df.interface == 'vfio') & (df.num_vms == 1) & (df.fastclick == 'hardware')].rxMppsCalc.mean()
    pt_software = df[(df.interface == 'vfio') & (df.num_vms == 1) & (df.fastclick == 'software')].rxMppsCalc.mean()
    percent = ((pt_rteflow - pt_software) / pt_software) * 100
    print(f"Rte_flow improves passthrough performance by +{percent:.0f}%.")

    mediation = df[(df.interface == 'vmux-med') & (df.num_vms == 1) & (df.fastclick == 'hardware')].rxMppsCalc.mean()
    print(f"vMux-med is {mediation:.2f} Mpps.")

    mediation = df[(df.interface == 'vmux-med') & (df.num_vms == 1) & (df.fastclick == 'hardware')].rxMppsCalc.mean()
    qemu_fastest = df[(df.interface == 'bridge') & (df.num_vms == 1) & (df.fastclick == 'software-tap')].rxMppsCalc.mean()
    qemu_slowest = df[(df.interface == 'bridge-e1000') & (df.num_vms == 1) & (df.fastclick == 'software-tap')].rxMppsCalc.mean()
    min_pct = ((mediation - qemu_fastest) / qemu_fastest) * 100
    max_pct = ((mediation - qemu_slowest) / qemu_slowest) * 100
    min_x = (mediation / qemu_fastest) - 1.0
    max_x = (mediation / qemu_slowest) - 1.0
    print(f"Mediation is +{min_pct:.0f}% to +{max_pct:.0f}% faster than qemu.")
    print(f"Mediation is {min_x:.1f}x to {max_x:.1f}x faster than qemu.")

    mediation = df[(df.interface == 'vmux-med') & (df.num_vms == 1) & (df.fastclick == 'hardware')].rxMppsCalc.mean()
    emulation = df[(df.interface == 'vmux-dpdk-e810') & (df.num_vms == 1) & (df.fastclick == 'hardware')].rxMppsCalc.mean()
    percent = ((emulation - mediation) / mediation) * 100
    print(f"Emulation of rte_flow is {-1*percent:.0f}% slower than mediation.")



if __name__ == '__main__':
    main()
