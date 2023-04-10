#!/bin/bash

q='normal'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do
        i='ioregionfdoff'
        r='xdp'
        for s in "60B" "1020B"; do
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

            load_latency_output="load_latency_${infix}.pdf"
            if [ ! -f "${load_latency_output}" ]; then
                echo "Plotting load latency for ${name}"

                ../../../../plot_load_latency.py \
                    --blue "../../../../dat/gr/virtio2/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                    --blue-name "${bluem}" \
                    --red "../../../../dat/gr/virtio2/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                    --red-name "${redm}" \
                    --logarithmic \
                    --compress \
                    --width 6 \
                    --height 3 \
                    --output "${load_latency_output}"
            fi

            packet_loss_output="packet_loss_${infix}.pdf"
            if [ ! -f "${packet_loss_output}" ]; then
                echo "Plotting packet loss for ${name}"

                ../../../../plot_packet_loss.py \
                    --blue "../../../../dat/gr/virtio2/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                    --blue-name "${bluem}" \
                    --red "../../../../dat/gr/virtio2/output_${redinfix1}_"*"_${redinfix2}_"* \
                    --red-name "${redm}" \
                    --width 6 \
                    --height 2 \
                    --output "${packet_loss_output}"
            fi
        done
    done
done
