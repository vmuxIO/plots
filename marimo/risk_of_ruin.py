import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy import stats
    from scipy.integrate import dblquad
    return dblquad, mo, np, pd, plt, sns, stats


@app.cell
def _():
    from math import sqrt

    def risk_of_ruin(s, mu, sigma):
        """
        s: starting capital
        mu: expected profit per round
        sigma: stddev of profit
        """
        r = sqrt(pow(mu, 2) + pow(sigma, 2))
        print(r)
        p_ruin = pow((2 / (1 + (mu / r))) - 1, s / r)
        print((2 / (1 + (mu / r))) - 1)
        return p_ruin

    # related : gambler's ruin with markov chains

    risk_of_ruin(10,0.1, 1)
    return risk_of_ruin, sqrt


@app.cell
def _(risk_of_ruin):
    # available pool resources
    s = 10
    mu = 0.00001
    sigma = 0.1
    risk_of_ruin(s, mu, sigma)
    return


@app.cell
def _():
    from scipy.stats import norm

    def p_shortage(occupancy, stddev, total):
        """
        P(shortage) = 1 - Roh((C - mu) / sigma)
        calculates the probability that a timestep or day has a shortage (i.e. breaks SLAs)
        """
        #occupancy = 90
        #stddev = 5
        #total = 100
        return float(1 - norm.cdf((total - occupancy) / stddev))

    p_shortage(occupancy=90, stddev=5, total=100)
    return (p_shortage,)


@app.cell
def _(p_shortage, sqrt):
    def _(nr_tenants = 10):
        tenant_mean = 1
        tenant_stddev = 5
        capacity_overhead = 0.5

        mean = nr_tenants * tenant_mean
        total_capacity = mean * ( 1 + 0.5 )
        stddev = sqrt(nr_tenants) * tenant_stddev
        shortage = p_shortage(mean, stddev, total_capacity)
        print(f"{nr_tenants} tenants ({mean} at {stddev})-> {shortage} shortage")
        return shortage


    _(nr_tenants=1000)
    _(nr_tenants=2000)
    return


@app.cell
def _(np, p_shortage, pd, plt, sns, sqrt):
    def _():
        # Parameters
        tenant_mean = 1
        tenant_stddev = 1
        capacity_overhead = 0.5
        n_tenants_range = np.arange(1, 101)

        # Calculate metrics for each number of tenants
        data = []
        for n in n_tenants_range:
            mean = n * tenant_mean
            stddev = sqrt(n) * tenant_stddev
            capacity = mean * (1 + capacity_overhead)
            # capacity = mean + 30
            shortage = p_shortage(mean, stddev, capacity)
            data.append({
                'n_tenants': n,
                'total_usage': mean,
                'total_capacity': capacity,
                'p_shortage': shortage
            })

        df = pd.DataFrame(data)

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot usage and capacity on left axis
        sns.lineplot(data=df, x='n_tenants', y='total_usage', ax=ax1, label='Total Usage (mean)', color='blue')
        sns.lineplot(data=df, x='n_tenants', y='total_capacity', ax=ax1, label='Total Capacity', color='green')

        # Plot p_shortage on right axis
        sns.lineplot(data=df, x='n_tenants', y='p_shortage', ax=ax2, label='P(shortage)', color='red')

        ax1.set_xlabel('Number of Tenants')
        ax1.set_ylabel('Servers', color='blue')
        ax2.set_ylabel('P(shortage)', color='red')
        # ax2.set_yscale('log')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title('Server Capacity Planning vs Number of Tenants')
        plt.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    By unifying VM pools, we reduce the physical resource capacity necessary to maintain a constant risk of violating SLAs:

    The bigger the pool, the greater the reduction in SLA violation risk from unifying two fragmented pools.
    """
    )
    return


@app.cell
def _(p_shortage, sqrt):
    def _():
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        # Parameters
        tenant_mean = 1
        tenant_stddev = 5
        nr_tenants_list = [1000, 2000, 10000, 20000, 100000, 200000]
        overhead_range = np.linspace(0, 1, 100)

        # Calculate p_shortage for each overhead and tenant count
        data = []
        for n in nr_tenants_list:
            mean = n * tenant_mean
            stddev = sqrt(n) * tenant_stddev
            for overhead in overhead_range:
                capacity = mean * (1 + overhead)
                shortage = p_shortage(mean, stddev, capacity)
                data.append({
                    'capacity_overhead': overhead,
                    'p_shortage': shortage,
                    'nr_tenants': n
                })

        df = pd.DataFrame(data)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x='capacity_overhead', y='p_shortage', hue='nr_tenants', ax=ax)

        ax.set_xlabel('Relative Capacity Overhead') # term is not intuitive
        ax.set_ylabel('P(shortage)')
        ax.set_yscale('log')

        plt.title('P(shortage) vs Capacity Overhead (tenant_mean=1, tenant_stddev=5)')
        plt.tight_layout()
        return fig


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The benefit of unified VM pools outweighs the performance penalty expected from downgrading VM networking (e.g.  from passthrough VMs to emulation). 
    Upgrades are of course always possible and require no further consideration. 

    The question boils down to: In our set of passthrough tenants, are there enough VMs that are cold enough to be transparently downgraded? Is it more likely that we violate SLAs because the passthrough tenant will require more resources or is it more likely that we violate SLAs because tenant usage outgrows our capacity?

    * 
    * We violate the SLA of the downgraded tenant when: tenant_mean > emulation_capacity
    * We can mitigate

    How much can we reduce SLA violation risk?

    (1) Enabling transparent upgrades: 

    (2) Enabling transparent downgrades
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Unified Pool Shortage Probability

    Let:

    - $X$ = EMU usage $\sim N(\mu_{emu}, \sigma_{emu})$
    - $Y$ = PT usage $\sim N(\mu_{pt}, \sigma_{pt})$
    - $X$ and $Y$ are independent
    - $C$ = resource capacity

    **Shortage conditions in unified pool:**

    - PT shortage: $Y > C_{pt}$ (PT always needs dedicated resources)
    - EMU shortage: $X > C_{emu} + \max(0, C_{pt} - Y)$ (EMU can use spare PT capacity)

    **Combined probability:**
    $$P(\text{shortage}) = P(Y > C_{pt}) + P(Y \leq C_{pt} \land X + Y > C_{emu} + C_{pt})$$

    The first term is PT overflow (EMU can't help). The second term is when total demand exceeds total capacity, but PT isn't individually over.
    """
    )
    return


@app.cell
def _(dblquad, np, p_shortage, stats):

    # TODO check if these functions correctly represent the markdown
    def p_shortage_unified(mu_emu, sigma_emu, cap_emu,
                           mu_pt, sigma_pt, cap_pt):
        """Exact calculation via numerical integration."""
        total_cap = cap_emu + cap_pt

        # P(Y > pt_cap) - PT overflow
        p_pt_overflow = 1 - stats.norm.cdf((cap_pt - mu_pt) / sigma_pt)

        # P(Y <= pt_cap AND X + Y > total_cap) via integration
        def integrand(x, y):
            px = stats.norm.pdf(x, mu_emu, sigma_emu)
            py = stats.norm.pdf(y, mu_pt, sigma_pt)
            return px * py

        # Integrate over region: y <= pt_cap AND x + y > total_cap
        p_emu_overflow, _ = dblquad(
            integrand,
            -np.inf, cap_pt,                    # y bounds
            lambda y: total_cap - y, np.inf     # x bounds given y
        )

        return p_pt_overflow + p_emu_overflow

    def p_shortage_unified_approx(mu_emu, sigma_emu, cap_emu,
                                   mu_pt, sigma_pt, cap_pt):
        """Upper bound approximation (simpler, slight overcount)."""
        # P(PT overflow)
        p_pt = 1 - stats.norm.cdf((cap_pt - mu_pt) / sigma_pt)

        # P(total overflow) - treat as single unified pool
        mu_total = mu_emu + mu_pt
        sigma_total = np.sqrt(sigma_emu**2 + sigma_pt**2)
        cap_total = cap_emu + cap_pt
        p_total = 1 - stats.norm.cdf((cap_total - mu_total) / sigma_total)

        return p_pt + p_total


    {
        "shortage": 2 * p_shortage(occupancy=1000, stddev=160,total=1500),
        "shortage unified": p_shortage_unified(mu_emu=1000, sigma_emu=160, cap_emu=1500, mu_pt=1000, sigma_pt=160, cap_pt=1500),
    }
    return


@app.cell
def _():
    from tqdm import tqdm
    import time

    # Test tqdm progress bar in marimo
    total = 0
    for i in tqdm(range(100), desc="Testing tqdm", ncols=120):
        time.sleep(0.01)
        total += i
    total
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With vMux, we can also use EMU capacity (if available) for PT VMs. However, PT workloads are 10x slower on EMU capacity, so we can only downgrade VMs that are approx.
    <10% utilized and we still break SLAs when the utilization of these VMs exceeds the capabilities of an EMU VM. What other distribution do we need to know? How to
    calculate this?

    We can't calculate this without temporal information on VM usage. Even having the utilization distribution per VM is insufficient, because utilization could be alternating between two extremes, breaking SLAs every second time-step. 

    Maybe we could plot how much downgrading reduces SLA violation probability over VM usage prediction accuracy.
    """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    spec = [
        ("VIRTIO_NET_F_CSUM ", "0", " Device handles packets with partial checksum offload."),
        ("VIRTIO_NET_F_GUEST_CSUM ", "1", " Driver handles packets with partial checksum."),
        ("VIRTIO_NET_F_CTRL_GUEST_OFFLOADS ", "2", " Control channel offloads reconfiguration support."),
        ("VIRTIO_NET_F_MTU", "3", " Device maximum MTU reporting is supported. If offered by the device, device advises driver about the value of its maximum MTU. If negotiated, the driver uses mtu as the maximum MTU value."),
        ("VIRTIO_NET_F_MAC ", "5", " Device has given MAC address."),
        ("VIRTIO_NET_F_GUEST_TSO4 ", "7", " Driver can receive TSOv4."),
        ("VIRTIO_NET_F_GUEST_TSO6 ", "8", " Driver can receive TSOv6."),
        ("VIRTIO_NET_F_GUEST_ECN ", "9", " Driver can receive TSO with ECN."),
        ("VIRTIO_NET_F_GUEST_UFO ", "10", " Driver can receive UFO."),
        ("VIRTIO_NET_F_HOST_TSO4 ", "11", " Device can receive TSOv4."),
        ("VIRTIO_NET_F_HOST_TSO6 ", "12", " Device can receive TSOv6."),
        ("VIRTIO_NET_F_HOST_ECN ", "13", " Device can receive TSO with ECN."),
        ("VIRTIO_NET_F_HOST_UFO ", "14", " Device can receive UFO."),
        ("VIRTIO_NET_F_MRG_RXBUF ", "15", " Driver can merge receive buffers."),
        ("VIRTIO_NET_F_STATUS ", "16", " Configuration status field is available."),
        ("VIRTIO_NET_F_CTRL_VQ ", "17", " Control channel is available."),
        ("VIRTIO_NET_F_CTRL_RX ", "18", " Control channel RX mode support."),
        ("VIRTIO_NET_F_CTRL_VLAN ", "19", " Control channel VLAN filtering."),
        ("VIRTIO_NET_F_CTRL_RX_EXTRA ", "20", " Control channel RX extra mode support."),
        ("VIRTIO_NET_F_GUEST_ANNOUNCE", "21", " Driver can send gratuitous packets."),
        ("VIRTIO_NET_F_MQ", "22", " Device supports multiqueue with automatic receive steering."),
        ("VIRTIO_NET_F_CTRL_MAC_ADDR", "23", " Set MAC address through control channel."),
        ("VIRTIO_NET_F_HASH_TUNNEL", "51", " Device supports inner header hash for encapsulated packets."),
        ("VIRTIO_NET_F_VQ_NOTF_COAL", "52", " Device supports virtqueue notification coalescing."),
        ("VIRTIO_NET_F_NOTF_COAL", "53", " Device supports notifications coalescing."),
        ("VIRTIO_NET_F_GUEST_USO4 ", "54", " Driver can receive USOv4 packets."),
        ("VIRTIO_NET_F_GUEST_USO6 ", "55", " Driver can receive USOv6 packets."),
        ("VIRTIO_NET_F_HOST_USO ", "56", " Device can receive USO packets. Unlike UFO (fragmenting the packet) the USO splits large UDP packet to several segments when each of these smaller packets has UDP header."),
        ("VIRTIO_NET_F_HASH_REPORT", "57", " Device can report per-packet hash value and a type of calculated hash."),
        ("VIRTIO_NET_F_GUEST_HDRLEN", "59", " Driver can provide the exact hdr_len value. Device benefits from knowing the exact header length."),
        ("VIRTIO_NET_F_RSS", "60", " Device supports RSS (receive-side scaling) with Toeplitz hash calculation and configurable hash parameters for receive steering."),
        ("VIRTIO_NET_F_RSC_EXT", "61", " Device can process duplicated ACKs and report number of coalesced segments and duplicated ACKs."),
        ("VIRTIO_NET_F_STANDBY", "62", " Device may act as a standby for a primary device with the same MAC address."),
        ("VIRTIO_NET_F_SPEED_DUPLEX", "63", " Device reports speed and duplex."),
    ]

    spec_df = pd.DataFrame(data={"bit": [ int(v) for k, v, d in spec ], "name": [ k for k, v, d in spec ], "description": [ d for k, v, d in spec ]})

    old   = "0110101001100000000000000000100000000000000000000000000000000000" # ecs.c5.large
    small = "1100011111111111111001000001100000000000000000000000000000000000" # ecs.e-c4m1.large
    biggg = "1100010111111011111001100000100011000000100000000000000000000000" # ecs.g8a.*

    def parse(spec_df, input, name):
        bits = [ idx for idx, c in enumerate(input) if c == '1' ]
        spec_df[name] = spec_df["bit"].isin(bits)
        return spec_df

    spec_df = parse(spec_df, old, "old")
    spec_df = parse(spec_df, small, "small")
    spec_df = parse(spec_df, biggg, "biggg")
    spec_df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Hypothesis:

    * Old generations (no Shenlong/CIPU) are limited to 0.3Mpps per queue, indicating software vSwitching.
        * Example: c5
        * Approx 35% are old and 65% are new 
        * Total: ~275, 30%
    * New generation use hardware-accelerated IO, achieving high Mpps. Example: g8a
        * Total: ~502, 56%
    * Economy uses cheapest emulation and achieves very poor throughput. Example: e-c4m1
        * Total: 74, 8%
    * Bare-metal:
        * Others can probably also do PT
        * Total: 49, 5%

    Found some proof, see Jan 14th 2026 "alibaba history". 
    Gen<=5 is software switch, gen6 is software switch with accel, gen7+ is CIPU. 

    Argument (approx numbers): 15% of instances use emulation to minimize cost, 10% offer passthrough for best performance, and 75% use CIPU slices.
    """
    )
    return


if __name__ == "__main__":
    app.run()
