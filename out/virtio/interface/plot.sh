#!/bin/bash

for m in 'pcvm' 'microvm'; do
    for q in 'normal' 'allregs' 'intstatus'; do
        if [ "$m" = "pcvm" ] && ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ]); then
            continue
        fi

        for v in 'vhostoff' 'vhoston'; do
            for i in 'ioregionfdoff' 'ioregionfdon'; do
                if [ "$m" = "pcvm" ] && [ "$i" = "ioregionfdon" ]; then
                    continue
                fi
                if ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ]) && [ "$i" = "ioregionfdoff" ]; then
                    continue
                fi
                if [ "$q" = "normal" ] && [ "$i" = "ioregionfdon" ]; then
                    continue
                fi

                for r in "moongen" "xdp"; do
                    if [ "$m" = "microvm" ] && [ "$r" = "moongen" ]; then
                        continue
                    fi

                    for s in "60B" "1020B"; do
                        name="$m $q $v $i $r $s"
                        infix="${m}_${q}_${v}_${i}_${r}_${s}"

                        bluen='bridge'
                        redn='macvtap'

                        bluename="$m $bluen $q $v $i $r $s"
                        redname="$m $redn $q $v $i $r $s"

                        blueinfix1="${m}_${bluen}_${q}_${v}_${i}_${r}"
                        blueinfix2="$s"
                        blueinfix="${blueinfix1}_${blueinfix2}"

                        redinfix1="${m}_${redn}_${q}_${v}_${i}_${r}"
                        redinfix2="$s"
                        redinfix="${redinfix1}_${redinfix2}"

                        echo "Plotting ${bluename} vs ${redname}"
                        # ls "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
                        # ls "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
                        # ls "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
                        # ls "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

                        ../../../plot_load_latency.py \
                            --blue "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                            --blue-name "${bluen}" \
                            --red "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                            --red-name "${redn}" \
                            --logarithmic \
                            --width 6 \
                            --height 6 \
                            --output "load_latency_${infix}.pdf"

                        ../../../plot_packet_loss.py \
                            --blue "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                            --blue-name "${bluen}" \
                            --red "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                            --red-name "${redn}" \
                            --width 6 \
                            --height 6 \
                            --output "packet_loss_${infix}.pdf"
                                done
                done
            done
        done
    done
done
