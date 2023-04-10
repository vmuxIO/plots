#!/bin/bash

m='microvm'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do

        r='xdp'
        for s in "60B" "1020B"; do
            name="$m $n $v $r $s"
            infix="${m}_${n}_${v}_${r}_${s}"

            blueq='normal'
            redq='intstatus'
            orangeq='allregs'

            bluei='ioregionfdoff'
            redi='ioregionfdon'
            orangei='ioregionfdon'

            bluename="$m $n $blueq $v $bluei $r $s"
            redname="$m $n $redq $v $redi $r $s"
            orangename="$m $n $orangeq $v $orangei $r $s"

            blueinfix1="${m}_${n}_${blueq}_${v}_${bluei}_${r}"
            blueinfix2="$s"
            blueinfix="${infix1}_${infix2}"

            redinfix1="${m}_${n}_${redq}_${v}_${redi}_${r}"
            redinfix2="$s"
            redinfix="${infix1}_${infix2}"

            orangeinfix1="${m}_${n}_${orangeq}_${v}_${orangei}_${r}"
            orangeinfix2="$s"
            orangeinfix="${infix1}_${infix2}"

            load_latency_output="load_latency_${infix}.pdf"
            if [ ! -f "${load_latency_output}" ]; then
                echo "Plotting load latency for ${name}"

                ../../../../plot_load_latency.py \
                    --blue "../../../../dat/gr/virtio2/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                    --blue-name "${blueq}" \
                    --red "../../../../dat/gr/virtio2/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                    --red-name "${redq}" \
                    --orange "../../../../dat/gr/virtio2/acc_histogram_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                    --orange-name "${orangeq}" \
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
                    --blue-name "${blueq}" \
                    --red "../../../../dat/gr/virtio2/output_${redinfix1}_"*"_${redinfix2}_"* \
                    --red-name "${redq}" \
                    --orange "../../../../dat/gr/virtio2/output_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                    --orange-name "${orangeq}" \
                    --width 6 \
                    --height 2 \
                    --output "${packet_loss_output}"
            fi
        done
    done
done
