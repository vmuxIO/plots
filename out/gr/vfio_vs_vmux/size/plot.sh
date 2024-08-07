#!/bin/bash

m='pcvm'
q='normal'
for n in 'vfio' 'vmux'; do
    v='vhostoff'
    i='ioregionfdoff'
    r='moongen'
    
    name="$m $n $q $v $i $r"
    infix="${m}_${n}_${q}_${v}_${i}_${r}"
    
    blacks='60B'
    blues='380B'
    greens='508B'
    yellows='600B'
    oranges='700B'
    reds='764B'
    violets='1020B'
    purples='1514B'
    # for s in '60B' '124B' '252B' '380B' '508B' '600B' '700B' '764B' '1020B' '1514B'; do
    
    blackname="$m $n $q $v $i $r $blacks"
    bluename="$m $n $q $v $i $r $blues"
    greenname="$m $n $q $v $i $r $greens"
    yellowname="$m $n $q $v $i $r $yellows"
    orangename="$m $n $q $v $i $r $oranges"
    redname="$m $n $q $v $i $r $reds"
    violetname="$m $n $q $v $i $r $violets"
    purplename="$m $n $q $v $i $r $purples"
    
    blackinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    blackinfix2="${blacks}"
    blackinfix="${blackinfix1}_${blackinfix2}"
    
    blueinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    blueinfix2="${blues}"
    blueinfix="${blueinfix1}_${blueinfix2}"
    
    greeninfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    greeninfix2="${greens}"
    greeninfix="${greeninfix1}_${greeninfix2}"
    
    yellowinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    yellowinfix2="${yellows}"
    yellowinfix="${yellowinfix1}_${yellowinfix2}"

    orangeinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    orangeinfix2="${oranges}"
    orangeinfix="${orangeinfix1}_${orangeinfix2}"
    
    redinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    redinfix2="${reds}"
    redinfix="${redinfix1}_${redinfix2}"

    violetinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    violetinfix2="${violets}"
    violetinfix="${violetinfix1}_${violetinfix2}"

    purpleinfix1="${m}_${n}_${q}_${v}_${i}_${r}"
    purpleinfix2="${purples}"
    purpleinfix="${purpleinfix1}_${purpleinfix2}"
    
    
    echo "Plotting ${name}"
    
    ../../../../plot_load_latency.py \
        --black "../../../../dat/vfio_vs_vmux2/acc_histogram_${blackinfix1}_"*"_${blackinfix2}_"* \
        --black-name "${blacks}" \
        --blue "../../../../dat/vfio_vs_vmux2/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
        --blue-name "${blues}" \
        --green "../../../../dat/vfio_vs_vmux2/acc_histogram_${greeninfix1}_"*"_${greeninfix2}_"* \
        --green-name "${greens}" \
        --yellow "../../../../dat/vfio_vs_vmux2/acc_histogram_${yellowinfix1}_"*"_${yellowinfix2}_"* \
        --yellow-name "${yellows}" \
        --orange "../../../../dat/vfio_vs_vmux2/acc_histogram_${orangeinfix1}_"*"_${orangeinfix2}_"* \
        --orange-name "${oranges}" \
        --red "../../../../dat/vfio_vs_vmux2/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
        --red-name "${reds}" \
        --violet "../../../../dat/vfio_vs_vmux2/acc_histogram_${violetinfix1}_"*"_${violetinfix2}_"* \
        --violet-name "${violets}" \
        --purple "../../../../dat/vfio_vs_vmux2/acc_histogram_${purpleinfix1}_"*"_${purpleinfix2}_"* \
        --purple-name "${purples}" \
        --logarithmic \
        --compress \
        --width 6 \
        --height 3 \
        --output "load_latency_${infix}.pdf"
    
    ../../../../plot_packet_loss.py \
        --black "../../../../dat/vfio_vs_vmux2/output_${blackinfix1}_"*"_${blackinfix2}_"* \
        --black-name "${blacks}" \
        --blue "../../../../dat/vfio_vs_vmux2/output_${blueinfix1}_"*"_${blueinfix2}_"* \
        --blue-name "${blues}" \
        --green "../../../../dat/vfio_vs_vmux2/output_${greeninfix1}_"*"_${greeninfix2}_"* \
        --green-name "${greens}" \
        --yellow "../../../../dat/vfio_vs_vmux2/output_${yellowinfix1}_"*"_${yellowinfix2}_"* \
        --yellow-name "${yellows}" \
        --orange "../../../../dat/vfio_vs_vmux2/output_${orangeinfix1}_"*"_${orangeinfix2}_"* \
        --orange-name "${oranges}" \
        --red "../../../../dat/vfio_vs_vmux2/output_${redinfix1}_"*"_${redinfix2}_"* \
        --red-name "${reds}" \
        --violet "../../../../dat/vfio_vs_vmux2/output_${violetinfix1}_"*"_${violetinfix2}_"* \
        --violet-name "${violets}" \
        --purple "../../../../dat/vfio_vs_vmux2/output_${purpleinfix1}_"*"_${purpleinfix2}_"* \
        --purple-name "${purples}" \
        --width 6 \
        --height 2 \
        --output "packet_loss_${infix}.pdf"
done
