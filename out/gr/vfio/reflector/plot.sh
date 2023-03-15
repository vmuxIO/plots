#!/bin/bash

m='pcvm'
for q in 'normal'; do
    n='vfio'
    v='vhostoff'
    i='ioregionfdoff'

    for s in "60B" "124B" "252B" "508B" "1020B"; do
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

        ../../../../plot_load_latency.py \
            --blue "../../../../dat/gr/vfio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
            --blue-name "${bluer}" \
            --red "../../../../dat/gr/vfio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
            --red-name "${redr}" \
            --logarithmic \
            --compress \
            --width 6 \
            --height 3 \
            --output "load_latency_${infix}.pdf"

        ../../../../plot_packet_loss.py \
            --blue "../../../../dat/gr/vfio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
            --blue-name "${bluer}" \
            --red "../../../../dat/gr/vfio/output_${redinfix1}_"*"_${redinfix2}_"* \
            --red-name "${redr}" \
            --width 6 \
            --height 2 \
            --output "packet_loss_${infix}.pdf"
    done
done
