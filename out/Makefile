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
		-W 3.3 -H 4.5 \
		--1-name "Qemu-pt" --1 $(DATA_DIR)/ycsb_*vms_vfio_-1rps_rep*.csv \
		--2-name "vMux-emu-e810" --2 $(DATA_DIR)/ycsb_*vms_vmux-dpdk-e810_-1rps_rep*.csv \
		--3-name "vMux-emu-e1000" --3 $(DATA_DIR)/ycsb_*vms_vmux-emu_-1rps_rep*.csv \
		--4-name "Qemu-e1000" --4 $(DATA_DIR)/ycsb_*vms_bridge-e1000_-1rps_rep*.csv \
		--5-name "Qemu-VirtIO" --5 $(DATA_DIR)/ycsb_*vms_bridge_-1rps_rep*.csv \
		--6-name "Qemu-vhost" --6 $(DATA_DIR)/ycsb_*vms_bridge-vhost_-1rps_rep*.csv \
		--7-name "vMux-med-e810" --7 $(DATA_DIR)/ycsb_*vms_vmux-med_-1rps_rep*.csv \

# vMux emu doesnt work and hence does not appear in the plot
microservices.pdf: $(DATA_FILES) ../plot_microservices.py
	python3 ../plot_microservices.py -W 3.3 -H 4.5 \
  	--1-name "media Qemu-virtio" --1 $(DATA_DIR)/mediaMicroservices_bridge_33vms_*rps_rep?.log \
  	--2-name "media Qemu-e1000" --2 $(DATA_DIR)/mediaMicroservices_bridge-e1000_33vms_*rps_rep?.log \
  	--3-name "media vMux-emu-e1000" --3 $(DATA_DIR)/mediaMicroservices_vmux-emu_33vms_*rps_rep?.log \
  	--4-name "media vMux-emu-e810" --4 $(DATA_DIR)/mediaMicroservices_vmux-dpdk-e810_33vms_*rps_rep?.log \
  	--5-name "media vMux-med-e810" --5 $(DATA_DIR)/mediaMicroservices_vmux-med_33vms_*rps_rep?.log \
  	--6-name "hotel Qemu-VirtIO" --6 $(DATA_DIR)/hotelReservation_bridge_19vms_*rps_rep?.log \
  	--7-name "hotel Qemu-e1000" --7 $(DATA_DIR)/hotelReservation_bridge-e1000_19vms_*rps_rep?.log \
  	--8-name "hotel vMux-emu-e1000" --8 $(DATA_DIR)/hotelReservation_vmux-emu_19vms_*rps_rep?.log \
  	--9-name "hotel vMux-emu-e810" --9 $(DATA_DIR)/hotelReservation_vmux-dpdk-e810_19vms_*rps_rep?.log \
  	--10-name "hotel vMux-med-e810" --10 $(DATA_DIR)/hotelReservation_vmux-med_19vms_*rps_rep?.log \
  	--11-name "social Qemu-VirtIO" --11 $(DATA_DIR)/socialNetwork_bridge_27vms_*rps_rep?.log \
  	--12-name "social Qemu-e1000" --12 $(DATA_DIR)/socialNetwork_bridge-e1000_27vms_*rps_rep?.log \
  	--13-name "social vMux-emu-e1000" --13 $(DATA_DIR)/socialNetwork_vmux-emu_27vms_*rps_rep?.log \
  	--14-name "social vMux-emu-e810" --14 $(DATA_DIR)/socialNetwork_vmux-dpdk-e810_27vms_*rps_rep?.log \
  	--15-name "social vMux-med-e810" --15 $(DATA_DIR)/socialNetwork_vmux-med_27vms_*rps_rep?.log \

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
	python3 ../plot_mediation.py -W 3.3 -H 4 -l \
  	-o mediation.pdf \
  	--1-name "vMux-med-e810 (w/ rte_flow)" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name "vMux-med-e810" --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-emu-e810 (w/ rte_flow)" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--4-name "vMux-emu-e810" --4 $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--5-name "vMux-pt (w/ rte_flow)" --5 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--6-name "vMux-pt" --6 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--7-name "Qemu-pt (w/ rte_flow)" --7 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	--8-name "Qemu-pt" --8 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \

mediation-slow.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W 3.3 -H 4 \
  	-o mediation-slow.pdf \
  	--1-name "vMux-med-e810 (w/ rte_flow)" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name "vMux-med-e810" --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-emu-e810 (w/ rte_flow)" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--4-name "vMux-emu-e810" --4 $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \

mediation-fast.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W 3.3 -H 4 \
  	-o mediation-fast.pdf \
  	--1-name "vMux-med-e810 (w/ rte_flow)" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-med-e810" --3 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--4-name "vMux-pt" --4 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--5-name "vMux-pt (w/ rte_flow)" --5 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--6-name "Qemu-pt" --6 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--7-name "Qemu-pt (w/ rte_flow)" --7 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	# --2-name "vMux-med-e810" --2 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_1000kpps_rep?.log \

microbenchmark-cdf.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W 3.3 -H 2 -l \
  	-o microbenchmark-cdf.pdf \
  	--1-name "vMux-pt" --1 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--2-name "Qemu-pt" --2 $(DATA_DIR)/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--3-name "Qemu-vhost" --3 ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--4-name "Qemu-virtio" --4 $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--5-name "Qemu-emu-e1000" --5 $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--6-name "vMux-emu-e1000" --6 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--7-name "vMux-emu-e810" --7 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--8-name "vMux-med-e810" --8 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \

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
	python3 ../plot_throughput_bars.py -W 5 -H 2.7 -l \
  	-o microbenchmark-bars.pdf \
  	--1-name "vMux-pt" --1 $(DATA_DIR)/autotest/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
  	--2-name "Qemu-pt" --2 $(DATA_DIR)/autotest/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
   	--3-name "Qemu-e1000" --3 $(DATA_DIR)/autotest/output_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--4-name "Qemu-VirtIO" --4 $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--5-name "Qemu-vhost" --5 $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--6-name "vMux-emu-e1000" --6 $(DATA_DIR)/autotest/output_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--7-name "vMux-emu-e810" --7 $(DATA_DIR)/autotest/output_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--8-name "vMux-med-e810" --8 $(DATA_DIR)/autotest/output_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \

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
