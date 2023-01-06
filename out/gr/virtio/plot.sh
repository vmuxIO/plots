#!/bin/bash

for m in 'pcvm' 'microvm'; do
    for q in 'normal' 'allregs' 'intstatus' 'postwrallregs' 'postwrintstatus'; do
        if [ "$m" = "pcvm" ] && ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ]); then
            continue
        fi

        for n in 'bridge' 'macvtap'; do
            for v in 'vhostoff' 'vhoston'; do
                for i in 'ioregionfdoff' 'ioregionfdon'; do
                    if [ "$m" = "pcvm" ] && [ "$i" = "ioregionfdon" ]; then
                        continue
                    fi
                    if ([ "$q" = "allregs" ] || [ "$q" = "intstatus" ] || [ "$q" = "postwrallregs" ] || [ "$q" = "postwrintstatus" ]) && [ "$i" = "ioregionfdoff" ]; then
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
                            if ([ "$q" = "postwrallregs" ] || [ "$q" = "postwrintstatus" ]) && [ "$s" = "1020B" ]; then
                                continue
                            fi
                            name="$m $n $q $v $i $r $s"
                            infix1="${m}_${n}_${q}_${v}_${i}_${r}"
                            infix2="$s"
                            infix="${infix1}_${infix2}"

                            load_latency_output="load_latency_${infix}.pdf"
                            if [ ! -f "${load_latency_output}" ]; then
                                echo "Plotting load latency for ${name}"

                                ../../../plot_load_latency.py \
                                    --red "../../../dat/gr/virtio/acc_histogram_${infix1}_"*"_${infix2}_"* \
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
                                    --red "../../../dat/gr/virtio/output_${infix1}_"*"_${infix2}_"* \
                                    --red-name "" \
                                    --width 6 \
                                    --height 6 \
                                    --output "${packet_loss_output}"
                            fi
                        done
                    done
                done
            done
        done
    done
done
