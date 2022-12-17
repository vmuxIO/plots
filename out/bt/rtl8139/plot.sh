#!/bin/bash

m='pcvm'
for n in 'bridge' 'macvtap'; do
    q='rtl8139'
    for v in 'vhostoff' 'vhoston'; do
        for i in 'ioregionfdoff' 'ioregionfdon'; do
            r='xdp'
            for s in "60B" "1020B"; do
                name="$m $n $q $v $i $r $s"
                infix1="${m}_${n}_${q}_${v}_${i}_${r}"
                infix2="$s"
                infix="${infix1}_${infix2}"

                echo "Plotting ${name}"
                # ls "../../../dat/bt/rtl8139/acc_histogram_${infix1}_"*"_${infix2}_"*
                # ls "../../../dat/bt/rtl8139/output_${infix1}_"*"_${infix2}_"*

                ../../../plot_load_latency.py \
                    --red "../../../dat/bt/rtl8139/acc_histogram_${infix1}_"*"_${infix2}_"* \
                    --red-name "" \
                    --logarithmic \
                    --width 6 \
                    --height 4 \
                    --output "load_latency_${infix}.pdf"

                ../../../plot_packet_loss.py \
                    --red "../../../dat/bt/rtl8139/output_${infix1}_"*"_${infix2}_"* \
                    --red-name "" \
                    --width 6 \
                    --height 4 \
                    --output "packet_loss_${infix}.pdf"
            done
        done
    done
done
