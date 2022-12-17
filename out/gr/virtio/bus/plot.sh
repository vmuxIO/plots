#!/bin/bash

q='normal'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do
        i='ioregionfdoff'
        r='xdp'
        for s in "60B"; do # "1020B"; do
            name="$n $q $v $i $r $s"
            infix="${n}_${q}_${v}_${i}_${r}_${s}"

            bluem='pcvm'
            redm='microvm'

            bluename="$bluem $name"
            redname="$redm $name"

            blueinfix1="${bluem}_${n}_${q}_${v}_${i}_${r}"
            blueinfix2="$s"
            blueinfix="${blueinfix1}_${blueinfix2}"

            redinfix1="${redm}_${n}_${q}_${v}_${i}_${r}"
            redinfix2="$s"
            redinfix="${redinfix1}_${redinfix2}"

            echo "Plotting ${name}"
            # ls "../../../../dat/gr/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../../dat/gr/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../../dat/gr/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
            # ls "../../../../dat/gr/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

            ../../../../plot_load_latency.py \
                --blue "../../../../dat/gr/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluem}" \
                --red "../../../../dat/gr/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redm}" \
                --logarithmic \
                --compress \
                --width 6 \
                --height 3 \
                --output "load_latency_${infix}.pdf"

            ../../../../plot_packet_loss.py \
                --blue "../../../../dat/gr/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluem}" \
                --red "../../../../dat/gr/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redm}" \
                --width 6 \
                --height 2 \
                --output "packet_loss_${infix}.pdf"
        done
    done
done
