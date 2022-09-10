#!/bin/bash

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_bridge_vhostoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_bridge_vhoston_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_macvtap_vhostoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_macvtap_vhoston_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_bridge_vhostoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_bridge_vhoston_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_macvtap_vhostoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "ioregionfdoff" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "ioregionfdon" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_macvtap_vhoston_xdp.pdf'
