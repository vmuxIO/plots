#!/bin/bash

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_bridge_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_bridge_vhoston_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_macvtap_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio_big_packets/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_macvtap_vhoston_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --output 'packet_loss_bridge_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/output_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --output 'packet_loss_bridge_vhoston_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --output 'packet_loss_macvtap_vhostoff_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio_big_packets/output_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --blue-name "pcvm" \
    --red ../../../dat/virtio_big_packets/output_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "microvm" \
    --width 6 --height 6 \
    --output 'packet_loss_macvtap_vhoston_ioregionfdoff_xdp.pdf'
