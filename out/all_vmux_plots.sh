set -x
python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  -o microbenchmark-cdf.pdf \
  --b-name "vMux-pt 10Mpps" --b ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_*s.csv \
  --r-name "Qemu-pt 10Mpps" --r ../dat/vmux0.0.5/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_*s.csv \
  --g-name "Qemu-vhost 10kpps" --g ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \
 --c-name "Qemu-virtio 10kpps" --c ../dat/vmux0.0.5/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  --y-name "vMux-emu 10kpps" --y ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  --k-name "Qemu-e1000 x" --k ../dat/vmux0.0.5/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv
  # --m-name "vMux-emu x" --m ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
# --c-name "Qemu-emu 10kpps" --c ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \



# --red-name "vfio wtf" --red dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_1020B_30s.csv \

python3 ../plot_throughput_bars.py -W 5 -H 2.6 -l \
  -o microbenchmark-bars.pdf \
  --blue-name "vMux-pt" --blue ../dat/vmux0.0.5/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
  --cyan-name "Qemu-pt" --cyan ../dat/vmux0.0.5/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
   --green-name "Qemu-virtio" --green ../dat/vmux0.0.5/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   --yellow-name "Qemu-e1000" --yellow ../dat/vmux0.0.5/output_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   --red-name "vMux-emu" --red ../dat/vmux0.0.5/output_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log

# # this (big packets) is boring
# python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
#   -o microbenchmark-bars-bigframes.pdf \
#   --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#   --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#    --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log
