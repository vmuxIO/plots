#!/bin/bash

for m in 'pcvm' 'microvm'; do
    for q in 'normal' 'allregs' 'intstatus' 'rtl8139'; do
        if [ "$m" = "microvm" ] && [ "$q" = "rtl8139" ]; then
            continue
        fi
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
                    if ([ "$q" = "normal" ] || [ "$q" = "rtl8139" ]) && [ "$i" = "ioregionfdon" ]; then
                        continue
                    fi

                    for r in "moongen" "xdp"; do
                        if [ "$m" = "microvm" ] && [ "$r" = "moongen" ]; then
                            continue
                        fi
                        if [ "$q" = "rtl8139" ] && [ "$r" = "moongen" ]; then
                            continue
                        fi

                        for s in "60B" "1020B"; do
                            name="$m $n $q $v $i $r $s"
                            infix1="${m}_${n}_${q}_${v}_${i}_${r}"
                            infix2="$s"

                            echo "Plotting ${name}"
                            ls "../../dat/virtio/acc_histogram_${infix1}_"*"_${infix2}_"*
                            ls "../../dat/virtio/output_${infix1}_"*"_${infix2}_"*

                            # ../../plot_load_latency.py --red "../../dat/virtio/acc_histogram_${infix}_"* --red-name "" --logarithmic --width 6 --height 6 --output "load_latency_${infix}.pdf"
                            # ../../plot_packet_loss.py --red "../../dat/virtio/output_${infix}_"* --red-name "" --width 6 --height 6 --output "packet_loss_${infix}.pdf"
                        done
                    done
                done
            done
        done
    done
done
