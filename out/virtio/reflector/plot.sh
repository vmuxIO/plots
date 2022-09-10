#!/bin/bash

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_bridge_vhostoff_ioregionfdoff.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_bridge_vhoston_ioregionfdoff.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_macvtap_vhostoff_ioregionfdoff.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_macvtap_vhoston_ioregionfdoff.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_bridge_vhostoff_ioregionfdoff.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_bridge_vhoston_ioregionfdoff.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_macvtap_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/output_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_macvtap_vhostoff_ioregionfdoff.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_macvtap_normal_vhoston_ioregionfdoff_moongen_* \
    --blue-name 'moongen' \
    --red ../../../dat/virtio/output_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name 'xdp' \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_macvtap_vhoston_ioregionfdoff.pdf'
