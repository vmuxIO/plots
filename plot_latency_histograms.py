#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

data_phy = np.genfromtxt('data/phy-hist.csv', delimiter=',')
data_brtap = np.genfromtxt('data/br-tap-hist.csv', delimiter=',')
data_macvtap = np.genfromtxt('data/macvtap-hist.csv', delimiter=',')

data_phy[:,0] = data_phy[:,0] / 1000000.
data_brtap[:,0] = data_brtap[:,0] / 1000000.
data_macvtap[:,0] = data_macvtap[:,0] / 1000000.

xmin = 0.
xmax = 1.1 * max(max(data_phy[:,0]), max(data_brtap[:,0]), max(data_macvtap[:,0]))
histrange = (xmin, xmax)

size_phy = int(sum(data_phy[:,1]))
size_brtap = int(sum(data_brtap[:,1]))
size_macvtap = int(sum(data_macvtap[:,1]))

weights_phy = np.ones_like(data_phy[:,1]) / len(data_phy[:,0])
weights_brtap = np.ones_like(data_brtap[:,1]) / len(data_brtap[:,0])
weights_macvtap = np.ones_like(data_macvtap[:,1]) / len(data_macvtap[:,0])

bins = 300

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
ax.set_axisbelow(True)
plt.title('Latency Histogram for different Network Interfaces')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')

ax.set_xticks(np.arange(xmin, xmax, 1.0))
ax.set_xticks(np.arange(xmin, xmax, 0.25), minor=True)
ax.set_yticks(np.arange(0., 1.1, 0.1))
ax.set_yticks(np.arange(0., 1.1, 0.025), minor=True)
plt.grid(which='major', alpha=0.5, linestyle='dotted', linewidth=0.5)
plt.grid(which='minor', alpha=0.2, linestyle='dotted', linewidth=0.5)

plt.hist(
    data_phy[:,0],
    range=histrange,
    weights=weights_phy,
    bins=bins,
    linewidth=0.5,
    edgecolor='black',
    label='Physical Intel 82599ES NIC',
)
plt.hist(
    data_brtap[:,0],
    range=histrange,
    weights=weights_brtap,
    bins=bins,
    linewidth=0.5,
    edgecolor='black',
    label='Bridged TAP virtio-net-pci Device',
)
plt.hist(
    data_macvtap[:,0],
    range=histrange,
    weights=weights_macvtap,
    bins=bins,
    linewidth=0.5,
    edgecolor='black',
    label='MacVTap virtio-net-pci Device',
)

legend = plt.legend()
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)
plt.savefig('plots/latency-histogram.pdf')
