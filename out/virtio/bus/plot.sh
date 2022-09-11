#!/bin/bash

q='normal'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do
        i='ioregionfdoff'
        r='xdp'
        for s in "60B" "1020B"; do
            name="$n $q $v $i $r $s"
            infix="${n}_${q}_${v}_${i}_${r}"

            bluem='pcvm'
            redm='microvm'

            bluename="$bluem $name"
            redname="$redm $name"

            blueinfix1="${bluem}_${infix}"
            blueinfix2="$s"
            blueinfix="${blueinfix1}_${blueinfix2}"

            redinfix1="${redm}_${infix}"
            redinfix2="$s"
            redinfix="${redinfix1}_${redinfix2}"

            echo "Plotting ${name}"
            # ls "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
            # ls "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

            ../../../plot_load_latency.py \
                --blue "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluem}" \
                --red "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redm}" \
                --logarithmic \
                --width 6 \
                --height 6 \
                --output "load_latency_${infix}.pdf"

            ../../../plot_packet_loss.py \
                --blue "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluem}" \
                --red "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redm}" \
                --width 6 \
                --height 6 \
                --output "packet_loss_${infix}.pdf"
        done
    done
done
