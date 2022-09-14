#!/bin/bash

m='pcvm'
n='bridge'
q='rtl8139'
v='vhostoff'
r='xdp'
s="60B"

name="$m $n $q $v $r $s"
infix="${m}_${n}_${q}_${v}_${r}_${s}"

bluei='ioregionfdoff'
redi='ioregionfdon'

bluename="$m $n $q $v $bluei $r $s"
redname="$m $n $q $v $redi $r $s"

blueinfix1="${m}_${n}_${q}_${v}_${bluei}_${r}"
blueinfix2="$s"
blueinfix="${blueinfix1}_${blueinfix2}"

redinfix1="${m}_${n}_${q}_${v}_${redi}_${r}"
redinfix2="$s"
redinfix="${redinfix1}_${redinfix2}"

echo "Plotting ${name}"
# ls "../../../dat/rtl8139-small-rates/acc_histogram_${infix1}_"*"_${infix2}_"*
# ls "../../../dat/rtl8139-small-rates/output_${infix1}_"*"_${infix2}_"*

../../../plot_load_latency.py \
    --blue "../../../dat/rtl8139-small-rates/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
    --blue-name "${bluei}" \
    --red "../../../dat/rtl8139-small-rates/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
    --red-name "${redi}" \
    --logarithmic \
    --width 6 \
    --height 6 \
    --output "load_latency_${infix}.pdf"

../../../plot_packet_loss.py \
    --blue "../../../dat/rtl8139-small-rates/output_${blueinfix1}_"*"_${blueinfix2}_"* \
    --blue-name "${bluei}" \
    --red "../../../dat/rtl8139-small-rates/output_${redinfix1}_"*"_${redinfix2}_"* \
    --red-name "${redi}" \
    --width 6 \
    --height 6 \
    --output "packet_loss_${infix}.pdf"
