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

avg_phy = np.average(data_phy[:,0], weights=weights_phy)
avg_brtap = np.average(data_brtap[:,0], weights=weights_brtap)
avg_macvtap = np.average(data_macvtap[:,0], weights=weights_macvtap)

def stddev(data, weights):
    return np.sqrt(np.average((data - np.average(data, weights=weights))**2, weights=weights))

stddev_phy = stddev(data_phy[:,0], weights=weights_phy)
stddev_brtap = stddev(data_brtap[:,0], weights=weights_brtap)
stddev_macvtap = stddev(data_macvtap[:,0], weights=weights_macvtap)

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
    facecolor='#ff5555',
    label='Physical Intel 82599ES NIC',
)
plt.hist(
    data_brtap[:,0],
    range=histrange,
    weights=weights_brtap,
    bins=bins,
    linewidth=0.5,
    edgecolor='black',
    facecolor='#55ff55',
    label='Bridged TAP virtio-net-pci Device',
)
plt.hist(
    data_macvtap[:,0],
    range=histrange,
    weights=weights_macvtap,
    bins=bins,
    linewidth=0.5,
    edgecolor='black',
    facecolor='#5555ff',
    label='MacVTap virtio-net-pci Device',
)

plt.axvline(
    x=avg_phy,
    color='#880000',
    linewidth=1.0,
    label=f'Average for Physical Intel 82599ES NIC: {avg_phy:.2f} ms',
)
plt.axvline(
    x=avg_brtap,
    color='#008800',
    linewidth=1.0,
    label=f'Average for Bridged TAP virtio-net-pci Device: {avg_brtap:.2f} ms',
)
plt.axvline(
    x=avg_macvtap,
    color='#000088',
    linewidth=1.0,
    label=f'Average for MacVTap virtio-net-pci Device: {avg_macvtap:.2f} ms',
)

plt.errorbar(
    avg_phy,
    0.5,
    xerr=stddev_phy,
    fmt='o',
    color='#880000',
    markersize=0,
    capsize=5,
    capthick=1,
    label=f'Std. Dev. for Physical Intel 82599ES NIC: {stddev_phy:.2f} ms',
)
plt.errorbar(
    avg_brtap,
    0.5,
    xerr=stddev_brtap,
    fmt='o',
    color='#008800',
    markersize=0,
    capsize=5,
    capthick=1,
    label=f'Std. Dev. for Bridged TAP virtio-net-pci Device: {stddev_brtap:.2f} ms',
)
plt.errorbar(
    avg_macvtap,
    0.5,
    xerr=stddev_macvtap,
    fmt='o',
    color='#000088',
    markersize=0,
    capsize=5,
    capthick=1,
    label=f'Std. Dev. for MacVTap virtio-net-pci Device: {stddev_macvtap:.2f} ms',
)

legend = plt.legend()
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.8)
plt.savefig('plots/latency-histogram.pdf')
