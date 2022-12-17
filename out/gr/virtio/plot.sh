#!/bin/bash

for m in 'pcvm' 'microvm'; do
    for q in 'normal'; do # 'allregs' 'intstatus'; do
        if [ "$m" = "pcvm" ] && ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ]); then
            continue
        fi

        for n in 'bridge' 'macvtap'; do
            for v in 'vhostoff' 'vhoston'; do
                for i in 'ioregionfdoff'; do # 'ioregionfdon'; do
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

                        for s in "60B"; do # "1020B"; do
                            name="$m $n $q $v $i $r $s"
                            infix1="${m}_${n}_${q}_${v}_${i}_${r}"
                            infix2="$s"
                            infix="${infix1}_${infix2}"

                            echo "Plotting ${name}"
                            # ls "../../../dat/gr/virtio/acc_histogram_${infix1}_"*"_${infix2}_"*
                            # ls "../../../dat/gr/virtio/output_${infix1}_"*"_${infix2}_"*

                            ../../../plot_load_latency.py \
                                --red "../../../dat/gr/virtio/acc_histogram_${infix1}_"*"_${infix2}_"* \
                                --red-name "" \
                                --logarithmic \
                                --width 6 \
                                --height 6 \
                                --output "load_latency_${infix}.pdf"

                            ../../../plot_packet_loss.py \
                                --red "../../../dat/gr/virtio/output_${infix1}_"*"_${infix2}_"* \
                                --red-name "" \
                                --width 6 \
                                --height 6 \
                                --output "packet_loss_${infix}.pdf"
                        done
                    done
                done
            done
        done
    done
done
