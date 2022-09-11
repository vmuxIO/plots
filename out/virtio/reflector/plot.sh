#!/bin/bash

m='pcvm'
q='normal'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do
        i='ioregionfdoff'
        for s in "60B" "1020B"; do
            name="$m $n $q $v $i $s"
            infix="${m}_${n}_${q}_${v}_${i}_${s}"

            bluer='moongen'
            redr='xdp'

            bluename="$m $n $q $v $i ${bluer} $s"
            redname="$m $n $q $v $i ${redr} $s"

            blueinfix1="${m}_${n}_${q}_${v}_${i}_${bluer}"
            blueinfix2="$s"
            blueinfix="${blueinfix1}_${blueinfix2}"

            redinfix1="${m}_${n}_${q}_${v}_${i}_${redr}"
            redinfix2="$s"
            redinfix="${redinfix1}_${redinfix2}"

            echo "Plotting ${name}"
            # ls "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
            # ls "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

            ../../../plot_load_latency.py \
                --blue "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluer}" \
                --red "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redr}" \
                --logarithmic \
                --width 6 \
                --height 4 \
                --output "load_latency_${infix}.pdf"

            ../../../plot_packet_loss.py \
                --blue "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluer}" \
                --red "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redr}" \
                --width 6 \
                --height 2 \
                --output "packet_loss_${infix}.pdf"
        done
    done
done
