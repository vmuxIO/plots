DATA_DIR := ../dat/vmux0.0.6/
DATA_FILES = $(wildcard $(DATA_DIR)/*)

all: mediation.pdf ycsb.pdf microservices.pdf iperf.pdf microbenchmark-cdf.pdf microbenchmark-bars.pdf

install:
	test -n "$(OVERLEAF)" # OVERLEAF must be set
	cp ycsb.pdf $(OVERLEAF)/ycsb.pdf
	cp microservices.pdf $(OVERLEAF)/microservices.pdf
	cp iperf.pdf $(OVERLEAF)/iperf.pdf
	cp mediation.pdf $(OVERLEAF)/mediation.pdf
	cp microbenchmark-cdf.pdf $(OVERLEAF)/microbenchmark-cdf.pdf
	cp microbenchmark-bars.pdf $(OVERLEAF)/microbenchmark-bars.pdf

ycsb.pdf: $(DATA_FILES) ../plot_ycsb.py
	python3 ../plot_ycsb.py -o ycsb.pdf \
		--blue-name qemu-virtio --blue $(DATA_DIR)/ycsb_*_-1rps_rep?.csv

microservices.pdf: $(DATA_FILES) ../plot_microservices.py
	python3 ../plot_microservices.py -W 3.3 -H 2 \
  	--blue-name "media qemu-virtio" --blue $(DATA_DIR)/mediaMicroservices_bridge_33vms_*rps_rep?.log \
  	--red-name "media qemu-e1000" --red $(DATA_DIR)/mediaMicroservices_bridge-e1000_33vms_*rps_rep?.log \
  	--green-name "media vmux-e1000" --green $(DATA_DIR)/mediaMicroservices_vmux-emu_33vms_*rps_rep?.log \
  	--salmon-name "media vmux-med-e810" --salmon $(DATA_DIR)/mediaMicroservices_vmux-med_33vms_*rps_rep?.log \
  	--cyan-name "hotel qemu-virtio" --cyan $(DATA_DIR)/hotelReservation_bridge_19vms_*rps_rep?.log \
  	--magenta-name "hotel qemu-e1000" --magenta $(DATA_DIR)/hotelReservation_bridge-e1000_19vms_*rps_rep?.log \
  	--yellow-name "hotel vmux-e1000" --yellow $(DATA_DIR)/hotelReservation_vmux-emu_19vms_*rps_rep?.log \
  	--firebrick-name "hotel vmux-med-e810" --firebrick $(DATA_DIR)/hotelReservation_vmux-med_19vms_*rps_rep?.log \
  	--wheat-name "social qemu-virtio" --wheat $(DATA_DIR)/socialNetwork_bridge_27vms_*rps_rep?.log \
  	--brown-name "social qemu-e1000" --brown $(DATA_DIR)/socialNetwork_bridge-e1000_27vms_*rps_rep?.log \
  	--orange-name "social vmux-e1000" --orange $(DATA_DIR)/socialNetwork_vmux-emu_27vms_*rps_rep?.log \
  	--violet-name "social vmux-med-e810" --violet $(DATA_DIR)/socialNetwork_vmux-med_27vms_*rps_rep?.log \

iperf.pdf: $(DATA_FILES) ../plot_iperf.py
	python3 ../plot_iperf.py -W 3.3 -H 2 \
  	-o iperf.pdf \
  	--blue-name foo --blue $(DATA_DIR)/iperf3_1vms_vfio_*_rep?.summary \
  	--red-name foo --red $(DATA_DIR)/iperf3_1vms_bridge-vhost_*_rep?.summary \
  	--green-name foo --green $(DATA_DIR)/iperf3_1vms_bridge_*_rep?.summary \
  	--cyan-name foo --cyan $(DATA_DIR)/iperf3_1vms_bridge-e1000_*_rep?.summary \
  	--yellow-name foo --yellow $(DATA_DIR)/iperf3_1vms_vmux-dpdk-e810_*_rep?.summary \
  	--magenta-name foo --magenta $(DATA_DIR)/iperf3_1vms_vmux-med_*_rep?.summary

#		--blue-name foo --blue $(DATA_DIR)/iperf3_1vms_vmux-pt_*_rep?.summary


mediation.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W 3.3 -H 2 -l \
  	-o mediation.pdf \
  	--yellow-name foo --yellow $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--orange-name bar --orange $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--yellow-name foo --yellow $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--green-name bar --green $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--brown-name bar --brown $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--wheat-name bar --wheat $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \

mediation-slow.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W 3.3 -H 2 \
  	-o mediation-slow.pdf \
  	--yellow-name foo --yellow $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--blue-name foo --blue $(DATA_DIR)/mediation_hardware_*vms_vmux-med_1000kpps_rep?.log \
  	--red-name bar --red $(DATA_DIR)/mediation_software_*vms_vmux-med_1000kpps_rep?.log \

mediation-fast.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W 3.3 -H 2 \
  	-o mediation-fast.pdf \
  	--yellow-name foo --yellow $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--blue-name foo --blue $(DATA_DIR)/mediation_hardware_*vms_vmux-med_1000kpps_rep?.log \
  	--red-name bar --red $(DATA_DIR)/mediation_software_*vms_vmux-med_1000kpps_rep?.log \
  	--green-name bar --green $(DATA_DIR)/mediation_software_*vms_vmux-pt_1000kpps_rep?.log \
  	--orange-name bar --orange $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_1000kpps_rep?.log \
  	--cyan-name bar --cyan $(DATA_DIR)/mediation_software_*vms_vfio_1000kpps_rep?.log \
  	--purple-name bar --purple $(DATA_DIR)/mediation_hardware_*vms_vfio_1000kpps_rep?.log \
  	--gray-name bar --gray $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	--violet-name bar --violet $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--brown-name bar --brown $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--wheat-name bar --wheat $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \

microbenchmark-cdf.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  	-o microbenchmark-cdf.pdf \
  	--blue-name "vMux-pt" --blue $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--red-name "Qemu-pt" --red $(DATA_DIR)/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--green-name "Qemu-vhost" --green ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--cyan-name "Qemu-virtio" --cyan $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--yellow-name "vMux-e1000" --yellow $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--brown-name "vMux-e810" --brown $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--orange-name "Qemu-e1000" --orange $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \

#  	--m-name "vMux-emu-e810 10kpps" --m $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv
#  	--m-name "vMux-spacing" --m ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv
#  	--m-name "vMux-emu x" --m ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
#  	--c-name "Qemu-emu 10kpps" --c ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \

me-cdf.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  	-o me-cdf.pdf \
  	--blue-name "Qemu-e1000 x" --blue ../dat/vmux0.0.5/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
  	--magenta-name "vMux-spacing" --magenta ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv \
  	--orange-name "hist1" --orange ./acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_90kpps_60B_60s.csv

#--r-name "hist2" --r ./hist2.csv

# --red-name "vfio wtf" --red dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_1020B_30s.csv \

microbenchmark-bars.pdf: $(DATA_FILES) ../plot_throughput_bars.py
	# TODO not enough measurement resolution for PT
	python3 ../plot_throughput_bars.py -W 5 -H 2.6 -l \
  	-o microbenchmark-bars.pdf \
  	--blue-name "vMux-pt" --blue $(DATA_DIR)/autotest/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
  	--cyan-name "Qemu-pt" --cyan $(DATA_DIR)/autotest/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
   	 --yellow-name "Qemu-e1000" --yellow $(DATA_DIR)/autotest/output_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	 --green-name "Qemu-virtio" --green $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	 --red-name "vMux-e1000-emu" --red $(DATA_DIR)/autotest/output_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	 --magenta-name "vMux-e810-emu" --magenta $(DATA_DIR)/autotest/output_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	 --orange-name "vMux-e810-med" --orange $(DATA_DIR)/autotest/output_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	 --violet-name "Qemu-vhost" --violet $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \

# # this (big packets) is boring
# python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
#   -o microbenchmark-bars-bigframes.pdf \
#   --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#   --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#    --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log

help:
	@echo "Make targets:"
	@echo "  - all"
	@echo "  - *.pdf"
	@echo "  - help"
	@echo "  - install"
	@echo "Note: you may use `make -B` to force rebuilding."
