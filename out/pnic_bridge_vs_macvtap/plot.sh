#!/bin/bash

../../plot_load_latency.py --blue ../../dat/pnic_bridge_vs_macvtap/acc_histogram_host_bridge_* --blue-name bridge --red ../../dat/pnic_bridge_vs_macvtap/acc_histogram_host_macvtap_* --red-name macvtap --logarithmic --output load_latency.pdf
../../plot_packet_loss.py --blue ../../dat/pnic_bridge_vs_macvtap/output_host_bridge_* --blue-name bridge --red ../../dat/pnic_bridge_vs_macvtap/output_host_macvtap_* --red-name macvtap --output packet_loss.pdf
