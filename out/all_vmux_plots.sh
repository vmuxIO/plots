set -x
set -e


python3 ../plot_ycsb.py --blue-name qemu-virtio --blue ../dat/output-multi-vm-4c3d2776/ycsb_*_-1rps_rep?.csv

python3 ../plot_microservices.py -W 3.3 -H 2 \
  --blue-name "media qemu-virtio" --blue ../dat/vmux0.0.6/mediaMicroservices_bridge_33vms_*rps_rep?.log \
  --red-name "media qemu-e1000" --red ../dat/vmux0.0.6/mediaMicroservices_bridge-e1000_33vms_*rps_rep?.log \
  --green-name "media vmux-e1000" --green ../dat/vmux0.0.6/mediaMicroservices_vmux-emu_33vms_*rps_rep?.log \
  --cyan-name "hotel qemu-virtio" --cyan ../dat/vmux0.0.6/hotelReservation_bridge_19vms_*rps_rep?.log \
  --magenta-name "hotel qemu-e1000" --magenta ../dat/vmux0.0.6/hotelReservation_bridge-e1000_19vms_*rps_rep?.log \
  --yellow-name "hotel vmux-e1000" --yellow ../dat/vmux0.0.6/hotelReservation_vmux-emu_19vms_*rps_rep?.log \
  --orange-name "social vmux-e1000" --orange ../dat/vmux0.0.6/socialNetwork_vmux-emu_27vms_*rps_rep?.log \


python3 ../plot_iperf.py -W 3.3 -H 2 \
  -o iperf.pdf \
  --blue-name foo --blue ../dat/vmux0.0.6/iperf3_1vms_vfio_*_rep?.summary \
  --red-name foo --red ../dat/vmux0.0.6/iperf3_1vms_bridge-vhost_*_rep?.summary \
  --green-name foo --green ../dat/vmux0.0.6/iperf3_1vms_bridge_*_rep?.summary \
  --cyan-name foo --cyan ../dat/vmux0.0.6/iperf3_1vms_bridge-e1000_*_rep?.summary \
  --yellow-name foo --yellow ../dat/vmux0.0.6/iperf3_1vms_vmux-dpdk-e810_*_rep?.summary \
  --magenta-name foo --magenta ../dat/vmux0.0.6/iperf3_1vms_vmux-med_*_rep?.summary
  # --blue-name foo --blue ../dat/vmux0.0.6/iperf3_1vms_vmux-pt_*_rep?.summary



python3 ../plot_mediation.py -W 3.3 -H 2 \
  -o mediation.pdf \
  --blue-name foo --blue ../dat/vmux0.0.6/mediation_hardware_*vms_vmux-med_rep?.log --red-name bar --red ../dat/vmux0.0.6/mediation_software_*vms_vmux-med_rep?.log



python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  -o microbenchmark-cdf.pdf \
  --b-name "vMux-pt 10Mpps" --b ../dat/vmux0.0.6/autotest/acc_histogram_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_*s.csv \
  --r-name "Qemu-pt 10Mpps" --r ../dat/vmux0.0.6/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_*s.csv \
  --g-name "Qemu-vhost 10kpps" --g ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \
 --c-name "Qemu-virtio 10kpps" --c ../dat/vmux0.0.6/autotest/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  --y-name "vMux-emu 10kpps" --y ../dat/vmux0.0.6/autotest/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  --k-name "Qemu-e1000 10kpps" --k ../dat/vmux0.0.6/autotest/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  --m-name "vMux-spacing" --m ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv
  # --m-name "vMux-emu x" --m ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
# --c-name "Qemu-emu 10kpps" --c ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \


python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  -o me-cdf.pdf \
  --k-name "Qemu-e1000 x" --k ../dat/vmux0.0.5/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
  --m-name "vMux-spacing" --m ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv \
  --b-name "hist1" --b ./acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_90kpps_60B_60s.csv

#--r-name "hist2" --r ./hist2.csv

# --red-name "vfio wtf" --red dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_1020B_30s.csv \

# TODO not enough measurement resolution for PT
python3 ../plot_throughput_bars.py -W 5 -H 2.6 -l \
  -o microbenchmark-bars.pdf \
  --blue-name "vMux-pt" --blue ../dat/vmux0.0.6/autotest/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
  --cyan-name "Qemu-pt" --cyan ../dat/vmux0.0.6/autotest/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
   --yellow-name "Qemu-e1000" --yellow ../dat/vmux0.0.6/autotest/output_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   --green-name "Qemu-virtio" --green ../dat/vmux0.0.6/autotest/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   --red-name "vMux-e1000-emu" --red ../dat/vmux0.0.6/autotest/output_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   --magenta-name "vMux-e810-emu" --magenta ../dat/vmux0.0.6/autotest/output_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log
   # --orange-name "vMux-e810-med" --magenta ../dat/vmux0.0.6/autotest/output_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log
   # --orange-name "Qemu-vhost" --orange ../dat/vmux0.0.6/autotest/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \

# # this (big packets) is boring
# python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
#   -o microbenchmark-bars-bigframes.pdf \
#   --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#   --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#    --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log
