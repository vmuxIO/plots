#!/bin/bash

# ../../plot_load_latency.py \
#     --gray ../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_moongen_* \
#     --gray-name "pcvm bridge vhostoff ioregionfdoff moongen" \
#     --brown ../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
#     --brown-name "pcvm bridge vhostoff ioregionfdoff xdp" \
#     --red ../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_* \
#     --red-name "pcvm bridge vhoston ioregionfdoff moongen" \
#     --saddlebrown ../../dat/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
#     --saddlebrown-name "pcvm bridge vhoston ioregionfdoff xdp" \
#     --orange ../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_moongen_* \
#     --orange-name "pcvm macvtap vhostoff ioregionfdoff moongen" \
#     --gold ../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
#     --gold-name "pcvm macvtap vhostoff ioregionfdoff xdp" \
#     --olive ../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_moongen_* \
#     --olive-name "pcvm macvtap vhoston ioregionfdoff moongen" \
#     --darkseagreen ../../dat/virtio/acc_histogram_pcvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
#     --darkseagreen-name "pcvm macvtap vhoston ioregionfdoff xdp" \
#     --green ../../dat/virtio/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdoff_xdp_* \
#     --green-name "microvm bridge vhostoff ioregionfdoff xdp" \
#     --lightseagreen ../../dat/virtio/acc_histogram_microvm_bridge_normal_vhostoff_ioregionfdon_xdp_* \
#     --lightseagreen-name "microvm bridge vhostoff ioregionfdon xdp" \
#     --cyan ../../dat/virtio/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdoff_xdp_* \
#     --cyan-name "microvm bridge vhoston ioregionfdoff xdp" \
#     --steelblue ../../dat/virtio/acc_histogram_microvm_bridge_normal_vhoston_ioregionfdon_xdp_* \
#     --steelblue-name "microvm bridge vhoston ioregionfdon xdp" \
#     --blue ../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdoff_xdp_* \
#     --blue-name "microvm macvtap vhostoff ioregionfdoff xdp" \
#     --indigo ../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhostoff_ioregionfdon_xdp_* \
#     --indigo-name "microvm macvtap vhostoff ioregionfdon xdp" \
#     --magenta ../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdoff_xdp_* \
#     --magenta-name "microvm macvtap vhoston ioregionfdoff xdp" \
#     --deeppink ../../dat/virtio/acc_histogram_microvm_macvtap_normal_vhoston_ioregionfdon_xdp_* \
#     --deeppink-name "microvm macvtap vhoston ioregionfdon xdp"

# pcvm bridge normal vhostoff ioregionfdoff moongen
# pcvm bridge normal vhostoff ioregionfdoff xdp
# pcvm bridge normal vhoston ioregionfdoff moongen
# pcvm bridge normal vhoston ioregionfdoff xdp
# pcvm macvtap normal vhostoff ioregionfdoff moongen
# pcvm macvtap normal vhostoff ioregionfdoff xdp
# pcvm macvtap normal vhoston ioregionfdoff moongen
# pcvm macvtap normal vhoston ioregionfdoff xdp
# microvm bridge normal vhostoff ioregionfdoff xdp
# microvm bridge normal vhostoff ioregionfdon xdp
# microvm bridge normal vhoston ioregionfdoff xdp
# microvm bridge normal vhoston ioregionfdon xdp
# microvm macvtap normal vhostoff ioregionfdoff xdp
# microvm macvtap normal vhostoff ioregionfdon xdp
# microvm macvtap normal vhoston ioregionfdoff xdp
# microvm macvtap normal vhoston ioregionfdon xdp

for m in 'pcvm' 'microvm'; do
    for n in 'bridge' 'macvtap'; do
        for v in 'vhostoff' 'vhoston'; do
            for i in 'ioregionfdoff' 'ioregionfdon'; do
                if [ "$m" = "pcvm" ] && [ "$i" = "ioregionfdon" ]; then
                    continue
                fi
                for r in "moongen" "xdp"; do
                    if [ "$m" = "microvm" ] && [ "$r" = "moongen" ]; then
                        continue
                    fi
                    name="$m $n normal $v $i $r"
                    infix="${name// /_}"
                    echo "Plotting ${name}"
                    ../../plot_load_latency.py --red "../../dat/virtio/acc_histogram_${infix}_"* --red-name "" --logarithmic --width 6 --height 6 --output "load_latency_${infix}.pdf"
                    ../../plot_packet_loss.py --red "../../dat/virtio/output_${infix}_"* --red-name "" --width 6 --height 6 --output "packet_loss_${infix}.pdf"
                done
            done
        done
    done
done
