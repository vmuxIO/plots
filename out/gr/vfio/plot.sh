#!/bin/bash

m='pcvm'
for q in 'normal'; do
    n='vfio'
    v='vhostoff'
    i='ioregionfdoff'

    for r in "moongen" "xdp"; do
        for s in "60B" "124B" "252B" "508B" "1020B"; do
            name="$m $n $q $v $i $r $s"
            infix1="${m}_${n}_${q}_${v}_${i}_${r}"
            infix2="$s"
            infix="${infix1}_${infix2}"

            load_latency_output="load_latency_${infix}.pdf"
            if [ ! -f "${load_latency_output}" ]; then
                echo "Plotting load latency for ${name}"

                ../../../plot_load_latency.py \
                    --red "../../../dat/gr/vfio/acc_histogram_${infix1}_"*"_${infix2}_"* \
                    --red-name "" \
                    --logarithmic \
                    --width 6 \
                    --height 6 \
                    --output "${load_latency_output}"
            fi

            packet_loss_output="packet_loss_${infix}.pdf"
            if [ ! -f "${packet_loss_output}" ]; then
                echo "Plotting packet loss for ${name}"

                ../../../plot_packet_loss.py \
                    --red "../../../dat/gr/vfio/output_${infix1}_"*"_${infix2}_"* \
                    --red-name "" \
                    --width 6 \
                    --height 6 \
                    --output "${packet_loss_output}"
            fi
        done
    done
done