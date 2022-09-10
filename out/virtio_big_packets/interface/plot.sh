#!/bin/bash

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_vhostoff_ioregionfdon_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_vhoston_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_vhoston_ioregionfdon_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_vhostoff_ioregionfdon_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_vhoston_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --blue-name "bridge" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "macvtap" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_vhoston_ioregionfdon_xdp.pdf'
