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

            echo "Plotting ${name}"
            # ls "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
            # ls "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
            # ls "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

            ../../../plot_load_latency.py \
                --blue "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${blueq} ${bluei}" \
                --red "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redq} ${redi}" \
                --orange "../../../dat/virtio/acc_histogram_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                --orange-name "${orangeq} ${orangei}" \
                --logarithmic \
                --width 6 \
                --height 4 \
                --output "load_latency_${infix}.pdf"

            ../../../plot_packet_loss.py \
                --blue "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${blueq} ${bluei}" \
                --red "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redq} ${redi}" \
                --orange "../../../dat/virtio/output_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                --orange-name "${orangeq} ${orangei}" \
                --width 6 \
                --height 2 \
                --output "packet_loss_${infix}.pdf"
        done
    done
done
