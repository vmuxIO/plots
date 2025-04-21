VERSION := vmux0.0.13
OUT_DIR := ../out/$(VERSION)/
# OUT_DIR := ./

DATA_DIR := ../dat/$(VERSION)/
DATA_FILES = $(wildcard $(DATA_DIR)/*)

PAPER_FIGURES := microbenchmark-cdf.pdf microbenchmark-bars.pdf mediation.pdf microservices.pdf ycsb.pdf packingtrace.pdf
# PAPER_FIGURES := mediation.pdf microservices.pdf ycsb.pdf
# PAPER_FIGURES := mediation.pdf iperf.pdf ycsb.pdf

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

all: $(PAPER_FIGURES)

install:
	test -n "$(OVERLEAF)" # OVERLEAF must be set
	for f in $(PAPER_FIGURES); do test -f $(OUT_DIR)/$$f && cp $(OUT_DIR)/$$f $(OVERLEAF)/$$f || true; done
	test -f $(OUT_DIR)/packingtrace_stranding.pdf && cp $(OUT_DIR)/packingtrace_stranding.pdf $(OVERLEAF)/packingtrace_stranding.pdf || true

packingtrace.pdf: $(wildcard ../dat/packingtrace/*) ../plot_packingtrace.py
	python3 ../plot_packingtrace.py -o $(OUT_DIR)/packingtrace.pdf \
		-W $(WIDTH2) -H 2.2 \
		--1-name "Unified" --1 ../dat/packingtrace/all_unified.checkpoint.output.pkl \
		--2-name "Fragmented" --2 ../dat/packingtrace/all_newfrag.checkpoint.output.pkl \
		--3-name "Migrate unified" --3 ../dat/packingtrace/migrate_unified3.checkpoint.samples.pkl \
		--4-name "Migrate fragmented" --4 ../dat/packingtrace/migrate_fragmented2.checkpoint.samples.pkl \

packingtrace_stranding.pdf: $(wildcard ../dat/packingtrace/*) ../plot_packingtrace.py
	python3 ../plot_packingtrace.py -o $(OUT_DIR)/packingtrace_stranding.pdf \
		--utilization \
		-W $(WIDTH2) -H 2.2 \
		--1-name "Unified" --1 ../dat/packingtrace/all_unified.checkpoint.output.pkl \
		--3-name "Fragmented" --3 ../dat/packingtrace/all_newfrag.checkpoint.output.pkl \

#	--2-name "Fragmented" --2 ../dat/packingtrace/all_fragmented.checkpoint.output.pkl \

ptp:
	python3 ../plot_ptp.py -o $(OUT_DIR)/ptp.pdf \
		-W $(WIDTH2) -H 2.2 \
		--1-name "vMux-med VM1 rep0" --1 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep0.log.vm1.filtered \
		--2-name "vMux-med VM1 rep1" --2 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep1.log.vm1.filtered \
		--3-name "vMux-med VM1 rep2" --3 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep2.log.vm1.filtered \
		--4-name "vMux-med VM2 rep0" --4 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep0.log.vm2.filtered \
		--5-name "vMux-med VM2 rep1" --5 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep1.log.vm2.filtered \
		--6-name "vMux-med VM2 rep2" --6 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep2.log.vm2.filtered \
		--7-name "Qemu-pt" --7 $(DATA_DIR)/./ptp_dpdk-guest_vfio_rep0.log.vm1.filtered \
		--8-name "Qemu-pt" --8 $(DATA_DIR)/./ptp_dpdk-guest_vfio_rep1.log.vm1.filtered \
		--9-name "Qemu-pt" --9 $(DATA_DIR)/./ptp_dpdk-guest_vfio_rep2.log.vm1.filtered \
		--10-name "Native" --10 $(DATA_DIR)/./ptp_dpdk-host_none_rep0.log.filtered \
		--11-name "Native" --11 $(DATA_DIR)/./ptp_dpdk-host_none_rep1.log.filtered \
		--12-name "Native" --12 $(DATA_DIR)/./ptp_dpdk-host_none_rep2.log.filtered \
		--13-name "vMux-emu" --13 $(DATA_DIR)/./ptp_dpdk-guest_vmux-dpdk-e810_rep0.log.vm1.filtered \
		--14-name "vMux-emu" --14 $(DATA_DIR)/./ptp_dpdk-guest_vmux-dpdk-e810_rep1.log.vm1.filtered \
		--15-name "vMux-emu" --15 $(DATA_DIR)/./ptp_dpdk-guest_vmux-dpdk-e810_rep2.log.vm1.filtered \
		--16-name "vMux-med VM1" --16 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep1.log.vm1.filtered $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep1.log.vm2.filtered \
		--17-name "vMux-med VM2" --17 $(DATA_DIR)/./ptp_dpdk-guest_vmux-med_rep?.log.vm2.filtered \
		--18-name "Qemu-pt" --18 $(DATA_DIR)/./ptp_dpdk-guest_vfio_rep?.log.vm1.filtered \
		--19-name "Native" --19 $(DATA_DIR)/./ptp_dpdk-host_none_rep?.log.filtered \
		--20-name "vMux-emu VM1" --20 $(DATA_DIR)/./ptp_dpdk-guest_vmux-dpdk-e810_rep?.log.vm1.filtered \
		--21-name "vMux-emu VM2" --21 $(DATA_DIR)/./ptp_dpdk-guest_vmux-dpdk-e810_rep?.log.vm2.filtered \
		--22-name "vMux-pt" --22 $(DATA_DIR)/./ptp_dpdk-guest_vmux-pt_rep?.log.vm1.filtered \


ycsb.pdf: $(DATA_FILES) ../plot_ycsb.py
	python3 ../plot_ycsb.py -o $(OUT_DIR)/ycsb.pdf \
		-W $(WIDTH2) -H 1.9 \
		--1-name "Qemu-pt" --1 $(DATA_DIR)/ycsb_*vms_vfio_-1rps_rep*.csv \
		--2-name "Qemu-e1000" --2 $(DATA_DIR)/ycsb_*vms_bridge-e1000_-1rps_rep*.csv \
		--3-name "Qemu-VirtIO" --3 $(DATA_DIR)/ycsb_*vms_bridge_-1rps_rep*.csv \
		--4-name "Qemu-vhost" --4 $(DATA_DIR)/ycsb_*vms_bridge-vhost_-1rps_rep*.csv \
		--5-name "vMux-emu-e1000" --5 $(DATA_DIR)/ycsb_*vms_vmux-emu_-1rps_rep*.csv \
		--6-name "vMux-emu-e810" --6 $(DATA_DIR)/ycsb_*vms_vmux-dpdk-e810_-1rps_rep*.csv \
		--7-name "vMux-med-e810" --7 $(DATA_DIR)/ycsb_*vms_vmux-med_-1rps_rep*.csv \
		--8-name "vMux-vDPDK" --8 $(DATA_DIR)/ycsb_*vms_vmux-vdpdk_-1rps_rep*.csv \

# vMux emu doesnt work and hence does not appear in the plot
microservices.pdf: $(DATA_FILES) ../plot_microservices.py
	python3 ../plot_microservices.py -o $(OUT_DIR)/microservices.pdf \
		-W $(DWIDTH2) -H 1.8 \
  	--1-name "Qemu-VirtIO" --1 $(DATA_DIR)/mediaMicroservices_bridge_33vms_*rps_rep?.log \
  	--2-name "Qemu-vhost" --2 $(DATA_DIR)/mediaMicroservices_bridge-vhost_33vms_*rps_rep?.log \
  	--3-name "Qemu-e1000" --3 $(DATA_DIR)/mediaMicroservices_bridge-e1000_33vms_*rps_rep?.log \
  	--4-name "vMux-emu-e1000" --4 $(DATA_DIR)/mediaMicroservices_vmux-emu_33vms_*rps_rep?.log \
  	--5-name "vMux-emu-e810" --5 $(DATA_DIR)/mediaMicroservices_vmux-dpdk-e810_33vms_*rps_rep?.log \
  	--6-name "vMux-med-e810" --6 $(DATA_DIR)/mediaMicroservices_vmux-med_33vms_*rps_rep?.log \
  	--7-name "Qemu-VirtIO" --7 $(DATA_DIR)/hotelReservation_bridge_19vms_*rps_rep?.log \
  	--8-name "Qemu-vhost" --8 $(DATA_DIR)/hotelReservation_bridge-vhost_19vms_*rps_rep?.log \
  	--9-name "Qemu-e1000" --9 $(DATA_DIR)/hotelReservation_bridge-e1000_19vms_*rps_rep?.log \
  	--10-name "vMux-emu-e1000" --10 $(DATA_DIR)/hotelReservation_vmux-emu_19vms_*rps_rep?.log \
  	--11-name "vMux-emu-e810" --11 $(DATA_DIR)/hotelReservation_vmux-dpdk-e810_19vms_*rps_rep?.log \
  	--12-name "vMux-med-e810" --12 $(DATA_DIR)/hotelReservation_vmux-med_19vms_*rps_rep?.log \
  	--13-name "Qemu-VirtIO" --13 $(DATA_DIR)/socialNetwork_bridge_27vms_*rps_rep?.log \
  	--14-name "Qemu-vhost" --14 $(DATA_DIR)/socialNetwork_bridge-vhost_27vms_*rps_rep?.log \
  	--15-name "Qemu-e1000" --15 $(DATA_DIR)/socialNetwork_bridge-e1000_27vms_*rps_rep?.log \
  	--16-name "vMux-emu-e1000" --16 $(DATA_DIR)/socialNetwork_vmux-emu_27vms_*rps_rep?.log \
  	--17-name "vMux-emu-e810" --17 $(DATA_DIR)/socialNetwork_vmux-dpdk-e810_27vms_*rps_rep?.log \
  	--18-name "vMux-med-e810" --18 $(DATA_DIR)/socialNetwork_vmux-med_27vms_*rps_rep?.log \
  	--19-name "vMux-vDPDK" --19 $(DATA_DIR)/mediaMicroservices_vmux-vdpdk_33vms_*rps_rep?.log \
  	--20-name "vMux-vDPDK" --20 $(DATA_DIR)/hotelReservation_vmux-vdpdk_19vms_*rps_rep?.log \
  	--21-name "vMux-vDPDK" --21 $(DATA_DIR)/socialNetwork_vmux-vdpdk_27vms_*rps_rep?.log \

slides-microservices.pdf: $(DATA_FILES) ../plot_microservices.py
	python3 ../plot_microservices.py -o $(OUT_DIR)/slides-microservices.pdf \
		-W $(DWIDTH2) -H 1.8 \
  	--2-name "Qemu emulation" --2 $(DATA_DIR)/mediaMicroservices_bridge-vhost_33vms_*rps_rep?.log \
  	--5-name "vMux emulation" --5 $(DATA_DIR)/mediaMicroservices_vmux-dpdk-e810_33vms_*rps_rep?.log \
  	--6-name "vMux mediation" --6 $(DATA_DIR)/mediaMicroservices_vmux-med_33vms_*rps_rep?.log \
  	--8-name "Qemu emulation" --8 $(DATA_DIR)/hotelReservation_bridge-vhost_19vms_*rps_rep?.log \
  	--11-name "vMux emulation" --11 $(DATA_DIR)/hotelReservation_vmux-dpdk-e810_19vms_*rps_rep?.log \
  	--12-name "vMux mediation" --12 $(DATA_DIR)/hotelReservation_vmux-med_19vms_*rps_rep?.log \
  	--14-name "Qemu emulation" --14 $(DATA_DIR)/socialNetwork_bridge-vhost_27vms_*rps_rep?.log \
  	--17-name "vMux emulation" --17 $(DATA_DIR)/socialNetwork_vmux-dpdk-e810_27vms_*rps_rep?.log \
  	--18-name "vMux mediation" --18 $(DATA_DIR)/socialNetwork_vmux-med_27vms_*rps_rep?.log \

iperf.pdf: $(DATA_FILES) ../plot_iperf.py
	python3 ../plot_iperf.py -W $(WIDTH) -H 2 \
  	-o $(OUT_DIR)/iperf.pdf \
  	--blue-name foo --blue $(DATA_DIR)/iperf3_*vms_vfio_forward_tcp_-1B_rep?.log \
  	--red-name foo --red $(DATA_DIR)/iperf3_*vms_bridge-vhost_forward_tcp_-1B_rep?.log \
  	--green-name foo --green $(DATA_DIR)/iperf3_*vms_bridge_forward_tcp_-1B_rep?.log \
  	--cyan-name foo --cyan $(DATA_DIR)/iperf3_*vms_bridge-e1000_forward_tcp_-1B_rep?.log \
  	--yellow-name foo --yellow $(DATA_DIR)/iperf3_*vms_vmux-dpdk-e810_forward_tcp_-1B_rep?.log \
  	--magenta-name foo --magenta $(DATA_DIR)/iperf3_*vms_vmux-med_forward_tcp_-1B_rep?.log

#		--blue-name foo --blue $(DATA_DIR)/iperf3_1vms_vmux-pt_*_rep?.summary


iperf-udp.pdf: $(DATA_FILES) ../plot_iperf.py
	python3 ../plot_iperf.py -W $(WIDTH) -H 2 \
  	-o $(OUT_DIR)/iperf-udp.pdf \
  	--blue-name foo --blue $(DATA_DIR)/iperf3_1vms_vfio_forward_udp_*B_rep?.log \
  	--yellow-name foo --yellow $(DATA_DIR)/iperf3_1vms_vmux-dpdk-e810_forward_udp_*B_rep?.log \
  	--magenta-name foo --magenta $(DATA_DIR)/iperf3_1vms_vmux-med_forward_udp_*B_rep?.log


mediation.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(DWIDTH) -H 2.25 -l \
  	-o $(OUT_DIR)/mediation.pdf \
  	--1-name "vMux-med-e810" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name 'vMux-med-e810 $$\ast$$ ' --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-emu-e810" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--4-name "vMux-emu-e810 $$\ast$$" --4 $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--5-name "Qemu-vhost $$\dag$$" --5 $(DATA_DIR)/mediation_software-tap_*vms_bridge-vhost_40000kpps_rep?.log \
  	--6-name "Qemu-VirtIO $$\dag$$" --6 $(DATA_DIR)/mediation_software-tap_*vms_bridge_40000kpps_rep?.log \
  	--7-name "Qemu-e1000 $$\dag$$" --7 $(DATA_DIR)/mediation_software-tap_*vms_bridge-e1000_40000kpps_rep?.log \
  	--8-name "vMux-pt" --8 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--9-name "vMux-pt $$\ast$$" --9 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--10-name "Qemu-pt" --10 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	--11-name "Qemu-pt $$\ast$$" --11 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--12-name "vMux-vDPDK" --12 $(DATA_DIR)/mediation_hardware_*vms_vmux-vdpdk_40000kpps_rep?.log \
  	--13-name "vMux-vDPDK $$\ast$$" --13 $(DATA_DIR)/mediation_software_*vms_vmux-vdpdk_40000kpps_rep?.log \

mediation1.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(WIDTH) -H 3 -l \
  	-o $(OUT_DIR)/mediation1.pdf \
  	--1-name "vMux-med-e810" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name 'vMux-med-e810 $$\ast$$ ' --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-emu-e810" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--4-name "vMux-emu-e810 $$\ast$$" --4 $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--5-name "Qemu-vhost $$\dag$$" --5 $(DATA_DIR)/mediation_software-tap_*vms_bridge-vhost_40000kpps_rep?.log \
  	--6-name "Qemu-VirtIO $$\dag$$" --6 $(DATA_DIR)/mediation_software-tap_*vms_bridge_40000kpps_rep?.log \
  	--7-name "Qemu-e1000 $$\dag$$" --7 $(DATA_DIR)/mediation_software-tap_*vms_bridge-e1000_40000kpps_rep?.log \

mediation2.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(WIDTH) -H 2.25 -l \
  	-o $(OUT_DIR)/mediation2.pdf \
  	--8-name "vMux-pt" --8 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--9-name "vMux-pt $$\ast$$" --9 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--10-name "Qemu-pt" --10 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	--11-name "Qemu-pt $$\ast$$" --11 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--12-name "vMux-vDPDK" --12 $(DATA_DIR)/mediation_hardware_*vms_vmux-vdpdk_40000kpps_rep?.log \
  	--13-name "vMux-vDPDK $$\ast$$" --13 $(DATA_DIR)/mediation_software_*vms_vmux-vdpdk_40000kpps_rep?.log \

slides-mediation.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(DWIDTH) -H 3 -l \
  	-o $(OUT_DIR)/slides-mediation.pdf \
  	-s \
  	--1-name "vMux mediation" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name 'vMux mediation $$\ast$$ ' --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux emulation" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--5-name "Qemu emulation $$\dag$$" --5 $(DATA_DIR)/mediation_software-tap_*vms_bridge-vhost_40000kpps_rep?.log \
  	--8-name "vMux passthrough" --8 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--9-name "vMux passthrough $$\ast$$" --9 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--10-name "Qemu passthrough" --10 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	--11-name "Qemu passthrough $$\ast$$" --11 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--12-name "vMux-vDPDK" --12 $(DATA_DIR)/mediation_hardware_*vms_vmux-vdpdk_40000kpps_rep?.log \
  	--13-name "vMux-vDPDK $$\ast$$" --13 $(DATA_DIR)/mediation_software_*vms_vmux-vdpdk_40000kpps_rep?.log \

mediation-slow.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(WIDTH) -H 4 \
  	-o $(OUT_DIR)/mediation-slow.pdf \
  	--1-name "vMux-med-e810 (w/ rte_flow)" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--2-name "vMux-med-e810" --2 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-emu-e810 (w/ rte_flow)" --3 $(DATA_DIR)/mediation_hardware_*vms_vmux-dpdk-e810_40000kpps_rep?.log \
  	--4-name "vMux-emu-e810" --4 $(DATA_DIR)/mediation_software_*vms_vmux-dpdk-e810_40000kpps_rep?.log \

mediation-fast.pdf: $(DATA_FILES) ../plot_mediation.py
	python3 ../plot_mediation.py -W $(WIDTH) -H 4 \
  	-o $(OUT_DIR)/mediation-fast.pdf \
  	--1-name "vMux-med-e810 (w/ rte_flow)" --1 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_40000kpps_rep?.log \
  	--3-name "vMux-med-e810" --3 $(DATA_DIR)/mediation_software_*vms_vmux-med_40000kpps_rep?.log \
  	--4-name "vMux-pt" --4 $(DATA_DIR)/mediation_software_*vms_vmux-pt_40000kpps_rep?.log \
  	--5-name "vMux-pt (w/ rte_flow)" --5 $(DATA_DIR)/mediation_hardware_*vms_vmux-pt_40000kpps_rep?.log \
  	--6-name "Qemu-pt" --6 $(DATA_DIR)/mediation_software_*vms_vfio_40000kpps_rep?.log \
  	--7-name "Qemu-pt (w/ rte_flow)" --7 $(DATA_DIR)/mediation_hardware_*vms_vfio_40000kpps_rep?.log \
  	# --2-name "vMux-med-e810" --2 $(DATA_DIR)/mediation_hardware_*vms_vmux-med_1000kpps_rep?.log \

microbenchmark-cdf.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W $(WIDTH) -H 2.1 -l \
  	-o $(OUT_DIR)/microbenchmark-cdf.pdf \
  	--1-name "vMux-pt" "l-" "blue" --1 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--2-name "vMux-vDPDK" "l--" "purple" --2 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-vdpdk_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--3-name "vMux-emu-e1000" "l--" "magenta" --3 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--4-name "vMux-emu-e810" "l--" "orange" --4 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--5-name "vMux-med-e810" "l-" "brown" --5 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--6-name "Qemu-pt" "l-." "red" --6 $(DATA_DIR)/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_*s.csv \
  	--7-name "Qemu-vhost" "l:" "green" --7 $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--8-name "Qemu-VirtIO" "l:" "cyan" --8 $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--9-name "Qemu-e1000" "l:" "violet" --9 $(DATA_DIR)/autotest/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \

#  	--m-name "vMux-emu-e810 10kpps" --m $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv
#  	--m-name "vMux-spacing" --m ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv
#  	--m-name "vMux-emu x" --m ../dat/vmux0.0.5/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
#  	--c-name "Qemu-emu 10kpps" --c ../dat/gr/virtio/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_30s.csv \


cdf-vmux-mg-vs-xdp.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W $(WIDTH) -H 2 -l \
  	-o $(OUT_DIR)/cdf-vmux-mg-vs-xdp.pdf \
  	--1-name "vMux-med-e810 xdp" "l--" "brown" --1 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--2-name "vMux-med-e810 moongen" "l:" "brown" --2 $(DATA_DIR)/autotest/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_30s.csv \
  	--3-name "Qemu-pt xdp" "l--" "orange" --3 $(DATA_DIR)/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_30s.csv \
  	--4-name "Qemu-pt moongen" "l:" "orange" --4 $(DATA_DIR)/autotest/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10kpps_60B_30s.csv \


me-cdf.pdf: $(DATA_FILES) ../plot_cdf.py
	python3 ../plot_cdf.py -W $(WIDTH) -H 2 -l \
  	-o $(OUT_DIR)/me-cdf.pdf \
  	--blue-name "Qemu-e1000 x" --blue ../dat/vmux0.0.5/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_30kpps_60B_*s.csv \
  	--magenta-name "vMux-spacing" --magenta ../dat/vmux-spacing/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_1000kpps_60B_*s.csv \
  	--orange-name "hist1" --orange ./acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_90kpps_60B_60s.csv

#--r-name "hist2" --r ./hist2.csv

# --red-name "vfio wtf" --red dat/vfio_vs_vmux2/acc_histogram_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_10000kpps_1020B_30s.csv \

microbenchmark-bars.pdf: $(DATA_FILES) ../plot_throughput_bars.py
	python3 ../plot_throughput_bars.py -W $(WIDTH) -H 2.4 -l \
  	-o $(OUT_DIR)/microbenchmark-bars.pdf \
  	--1-name "vMux-pt" --1 $(DATA_DIR)/autotest/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
  	--2-name "Qemu-pt" --2 $(DATA_DIR)/autotest/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
   	--3-name "Qemu-e1000" --3 $(DATA_DIR)/autotest/output_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--4-name "Qemu-VirtIO" --4 $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--5-name "Qemu-vhost" --5 $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--6-name "vMux-emu-e1000" --6 $(DATA_DIR)/autotest/output_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--7-name "vMux-emu-e810" --7 $(DATA_DIR)/autotest/output_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--8-name "vMux-med-e810" --8 $(DATA_DIR)/autotest/output_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
   	--9-name "vMux-vDPDK" --9 $(DATA_DIR)/autotest/output_pcvm_vmux-vdpdk_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \

slides-microbenchmark-bars.pdf: $(DATA_FILES) ../plot_throughput_bars.py
	python3 ../plot_throughput_bars.py -W $(WIDTH) -H 2.4 -l \
    -s \
    -o $(OUT_DIR)/slides-microbenchmark-bars.pdf \
    --1-name "vMux passthrough" --1 $(DATA_DIR)/autotest/output_pcvm_vmux-pt_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
    --2-name "Qemu passthrough" --2 $(DATA_DIR)/autotest/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \
    --5-name "Qemu emulation" --5 $(DATA_DIR)/autotest/output_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
    --7-name "vMux emulation" --7 $(DATA_DIR)/autotest/output_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
    --8-name "vMux mediation" --8 $(DATA_DIR)/autotest/output_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_*kpps_60B_*s_rep*.log \
    --9-name "vMux-vDPDK" --9 $(DATA_DIR)/autotest/output_pcvm_vmux-vdpdk_normal_vhostoff_ioregionfdoff_moongen_*kpps_60B_*s_rep*.log \

# # this (big packets) is boring
# python3 ../plot_throughput_bars.py -W 3.3 -H 2 -l \
#   -o microbenchmark-bars-bigframes.pdf \
#   --blue-name "vMux" --blue ../dat/vfio_vs_vmux2/output_pcvm_vmux_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#   --cyan-name "vfio" --cyan ../dat/vfio_vs_vmux2/output_pcvm_vfio_normal_vhostoff_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log \
#    --green-name "vhost" --green ../dat/gr/virtio2/output_pcvm_bridge_normal_vhoston_ioregionfdoff_moongen_*kpps_1020B_30s_rep*.log

vmtype-zones.pdf: $(DATA_FILES) ../plot_vmtype_zones.py
	python3 ../plot_vmtype_zones.py -W $(WIDTH) -H 1.8 \
    -o $(OUT_DIR)/vmtype-zones.pdf

help:
	@echo "Make targets:"
	@echo "  - all"
	@echo "  - *.pdf"
	@echo "  - help"
	@echo "  - install"
	@echo "Note: you may use `make -B` to force rebuilding."
