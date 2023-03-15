#!/bin/bash

m='pcvm'
for q in 'normal'; do
    n='vfio'
    v='vhostoff'
    i='ioregionfdoff'
    for r in "moongen" "xdp"; do
        name="$m $n $q $v $i $r"
        infix="${m}_${n}_${q}_${v}_${i}_${r}"

        blues='60B'
        greens='124B'
        oranges='252B'
        purples='508B'
        reds='1020B'

        bluename="$m $n $q $v $i $r $blues"
        greenname="$m $n $q $v $i $r $greens"
        orangename="$m $n $q $v $i $r $oranges"
        purplename="$m $n $q $v $i $r $purples"
        redname="$m $n $q $v $i $r $reds"
        

        blueinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
        blueinfix2="${blues}"
        blueinfix="${blueinfix1}_${blueinfix2}"

        greeninfix1="${m}_${n}_${q}_${v}_${i}_${r}"
        greeninfix2="${greens}"
        greeninfix="${greeninfix1}_${greeninfix2}"

        orangeinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
        orangeinfix2="${oranges}"
        orangeinfix="${orangeinfix1}_${orangeinfix2}"

        purpleinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
        purpleinfix2="${purples}"
        purpleinfix="${purpleinfix1}_${purpleinfix2}"

        redinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
        redinfix2="${reds}"
        redinfix="${redinfix1}_${redinfix2}"

        echo "Plotting ${name}"

        ../../../../plot_load_latency.py \
            --blue "../../../../dat/gr/vfio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
            --blue-name "${blues}" \
            --green "../../../../dat/gr/vfio/acc_histogram_${greeninfix1}_"*"_${greeninfix2}_"* \
            --green-name "${greens}" \
            --orange "../../../../dat/gr/vfio/acc_histogram_${orangeinfix1}_"*"_${orangeinfix2}_"* \
            --orange-name "${oranges}" \
            --purple "../../../../dat/gr/vfio/acc_histogram_${purpleinfix1}_"*"_${purpleinfix2}_"* \
            --purple-name "${purples}" \
            --red "../../../../dat/gr/vfio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
            --red-name "${reds}" \
            --logarithmic \
            --compress \
            --width 6 \
            --height 3 \
            --output "load_latency_${infix}.pdf"

        ../../../../plot_packet_loss.py \
            --blue "../../../../dat/gr/vfio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
            --blue-name "${blues}" \
            --green "../../../../dat/gr/vfio/output_${greeninfix1}_"*"_${greeninfix2}_"* \
            --green-name "${greens}" \
            --orange "../../../../dat/gr/vfio/output_${orangeinfix1}_"*"_${orangeinfix2}_"* \
            --orange-name "${oranges}" \
            --purple "../../../../dat/gr/vfio/output_${purpleinfix1}_"*"_${purpleinfix2}_"* \
            --purple-name "${purples}" \
            --red "../../../../dat/gr/vfio/output_${redinfix1}_"*"_${redinfix2}_"* \
            --red-name "${reds}" \
            --width 6 \
            --height 2 \
            --output "packet_loss_${infix}.pdf"
    done
done
