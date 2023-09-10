set -x
python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  -o microbenchmark-cdf.pdf \
  --b-name "vmux 10Mpps" --b ../dat/vfio_vs_vmux2/acc_histogram_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_30s.csv \
  --r-name "vfio 10Mpps" --r ../dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_60B_30s.csv \
  --g-name "vhost 20kpps" --g ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_20kpps_60B_30s.csv \
  --c-name "vhost 10kpps" --c ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv

# --red-name "vfio wtf" --red dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_1020B_30s.csv \

python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
  -o microbenchmark-bars.pdf \
  --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_30s_rep*.log \
  --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_30s_rep*.log \
   --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_60B_30s_rep*.log

# # this (big packets) is boring
# python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
#   -o microbenchmark-bars-bigframes.pdf \
#   --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#   --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#    --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log
