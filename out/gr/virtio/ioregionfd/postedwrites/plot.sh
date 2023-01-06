#!/bin/bash

m='microvm'
for n in 'bridge' 'macvtap'; do
    for v in 'vhostoff' 'vhoston'; do

        r='xdp'
        s='60B'

        name="$m $n $v $r $s"
        infix="${m}_${n}_${v}_${r}_${s}"

        blueq='normal'
        redq='intstatus'
        orangeq='postwrintstatus'
        darkgreenq='allregs'
        limegreenq='postwrallregs'

        bluelabel='normal'
        redlabel='intstatus'
        orangelabel='postint'
        darkgreenlabel='allregs'
        limegreenlabel='postall'

        bluei='ioregionfdoff'
        redi='ioregionfdon'
        orangei='ioregionfdon'
        darkgreeni='ioregionfdon'
        limegreeni='ioregionfdon'

        bluename="$m $n $blueq $v $bluei $r $s"
        redname="$m $n $redq $v $redi $r $s"
        orangename="$m $n $orangeq $v $orangei $r $s"
        darkgreenname="$m $n $darkgreenq $v $darkgreeni $r $s"
        limegreenname="$m $n $limegreenq $v $limegreeni $r $s"

        blueinfix1="${m}_${n}_${blueq}_${v}_${bluei}_${r}"
        blueinfix2="$s"
        blueinfix="${infix1}_${infix2}"

        redinfix1="${m}_${n}_${redq}_${v}_${redi}_${r}"
        redinfix2="$s"
        redinfix="${infix1}_${infix2}"

        orangeinfix1="${m}_${n}_${orangeq}_${v}_${orangei}_${r}"
        orangeinfix2="$s"
        orangeinfix="${infix1}_${infix2}"

        darkgreeninfix1="${m}_${n}_${darkgreenq}_${v}_${darkgreeni}_${r}"
        darkgreeninfix2="$s"
        darkgreeninfix="${infix1}_${infix2}"

        limegreeninfix1="${m}_${n}_${limegreenq}_${v}_${limegreeni}_${r}"
        limegreeninfix2="$s"
        limegreeninfix="${infix1}_${infix2}"

        load_latency_output="load_latency_${infix}.pdf"
        if [ ! -f "${load_latency_output}" ]; then
            echo "Plotting load latency for ${name}"

            ../../../../../plot_load_latency.py \
                --blue "../../../../../dat/gr/virtio/acc_histogram_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluelabel}" \
                --red "../../../../../dat/gr/virtio/acc_histogram_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redlabel}" \
                --orange "../../../../../dat/gr/virtio/acc_histogram_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                --orange-name "${orangelabel}" \
                --darkgreen "../../../../../dat/gr/virtio/acc_histogram_${darkgreeninfix1}_"*"_${darkgreeninfix2}_"* \
                --darkgreen-name "${darkgreenlabel}" \
                --limegreen "../../../../../dat/gr/virtio/acc_histogram_${limegreeninfix1}_"*"_${limegreeninfix2}_"* \
                --limegreen-name "${limegreenlabel}" \
                --logarithmic \
                --compress \
                --width 9 \
                --height 3 \
                --output "${load_latency_output}"
        fi

        packet_loss_output="packet_loss_${infix}.pdf"
        if [ ! -f "${packet_loss_output}" ]; then
            echo "Plotting packet loss for ${name}"

            ../../../../../plot_packet_loss.py \
                --blue "../../../../../dat/gr/virtio/output_${blueinfix1}_"*"_${blueinfix2}_"* \
                --blue-name "${bluelabel}" \
                --red "../../../../../dat/gr/virtio/output_${redinfix1}_"*"_${redinfix2}_"* \
                --red-name "${redlabel}" \
                --orange "../../../../../dat/gr/virtio/output_${orangeinfix1}_"*"_${orangeinfix2}_"* \
                --orange-name "${orangelabel}" \
                --darkgreen "../../../../../dat/gr/virtio/output_${darkgreeninfix1}_"*"_${darkgreeninfix2}_"* \
                --darkgreen-name "${darkgreenlabel}" \
                --limegreen "../../../../../dat/gr/virtio/output_${limegreeninfix1}_"*"_${limegreeninfix2}_"* \
                --limegreen-name "${limegreenlabel}" \
                --width 9 \
                --height 2 \
                --output "${packet_loss_output}"
        fi
    done
done
