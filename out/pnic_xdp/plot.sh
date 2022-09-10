#!/bin/bash

../../plot_load_latency.py --red ../../dat/pnic_xdp/acc_* --red-name 'pnic xdp' --logarithmic --width 6 --height 4 --output load_latency.pdf
../../plot_packet_loss.py --red ../../dat/pnic_xdp/output_* --red-name 'pnic xdp' --width 6 --height 4 --output packet_loss.pdf
