#!/bin/bash

../../plot_load_latency.py --blue ../../dat/bt/pnic_raw/acc_* --blue-name 'pnic dpdk' --width 6 --height 4 --output load_latency.pdf
../../plot_packet_loss.py --blue ../../dat/bt/pnic_raw/output_* --blue-name 'pnic dpdk' --width 6 --height 4 --output packet_loss.pdf
