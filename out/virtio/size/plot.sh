#!/bin/bash

for m in 'pcvm' 'microvm'; do
    for q in 'normal' 'allregs' 'intstatus'; do
        if [ "$m" = "pcvm" ] && ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ]); then
            continue
        fi

        for n in 'bridge' 'macvtap'; do
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

                        name="$m $n $q $v $i $r $s"
                        infix="${m}_${n}_${q}_${v}_${i}_${r}"

                        blues="60B"
                        reds="1020B"

                        blueinfix1="${infix}"
                        blueinfix2="$blues"
                        blueinfix="${blueinfix1}_${blueinfix2}"

                        redinfix1="${infix}"
                        redinfix2="$reds"
                        redinfix="${redinfix1}_${redinfix2}"

                        echo "Plotting ${name}"
                        # ls "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"*
                        # ls "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"*
                        # ls "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"*
                        # ls "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"*

                        ../../../plot_load_latency.py \
                            --blue "../../../dat/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                            --blue-name "${blues}" \
                            --red "../../../dat/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                            --red-name "${reds}" \
                            --logarithmic \
                            --width 6 \
                            --height 6 \
                            --output "load_latency_${infix}.pdf"

                        ../../../plot_packet_loss.py \
                            --blue "../../../dat/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                            --blue-name "${blues}" \
                            --red "../../../dat/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                            --red-name "${reds}" \
                            --width 6 \
                            --height 6 \
                            --output "packet_loss_${infix}.pdf"
                    done
                done
            done
        done
    done
done
