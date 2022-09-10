#!/bin/bash

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_bridge_ioregionfdoff_moongen.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_bridge_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_moongen_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_macvtap_ioregionfdoff_moongen.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_pcvm_macvtap_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_bridge_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_bridge_ioregionfdon_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_macvtap_ioregionfdoff_xdp.pdf'

../../../plot_load_latency.py \
    --blue ../../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --logarithmic \
    --output 'load_latency_microvm_macvtap_ioregionfdon_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_bridge_ioregionfdoff_moongen.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_bridge_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_macvtap_normal_vhostoff_ioregionfdoff_moongen_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_pcvm_macvtap_normal_vhoston_ioregionfdoff_moongen_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_macvtap_ioregionfdoff_moongen.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_pcvm_macvtap_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_bridge_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_bridge_ioregionfdon_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_macvtap_ioregionfdoff_xdp.pdf'

../../../plot_packet_loss.py \
    --blue ../../../dat/virtio/output_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
    --blue-name "vhostoff" \
    --red ../../../dat/virtio/output_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
    --red-name "vhoston" \
    --width 6 --height 6 \
    --output 'packet_loss_microvm_macvtap_ioregionfdon_xdp.pdf'
