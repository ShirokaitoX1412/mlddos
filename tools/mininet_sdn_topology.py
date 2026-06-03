"""Mininet topology for the ML DDoS SDN demo.

Run on Ubuntu:
    sudo python3 tools/mininet_sdn_topology.py
"""

from __future__ import annotations

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo


class DDoSDemoTopo(Topo):
    """Single-switch SDN lab with one victim, one benign client, one attacker."""

    def build(self) -> None:
        switch = self.addSwitch("s1", protocols="OpenFlow13")

        benign = self.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
        victim = self.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")
        attacker = self.addHost("h3", ip="10.0.0.3/24", mac="00:00:00:00:00:03")

        self.addLink(benign, switch, cls=TCLink, bw=20, delay="2ms")
        self.addLink(victim, switch, cls=TCLink, bw=20, delay="2ms")
        self.addLink(attacker, switch, cls=TCLink, bw=20, delay="2ms")


def run() -> None:
    topo = DDoSDemoTopo()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True,
    )
    net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653,
    )

    net.start()
    print("\nSDN DDoS demo topology is running.")
    print("Hosts:")
    print("  h1 benign client  10.0.0.1")
    print("  h2 victim server  10.0.0.2")
    print("  h3 attacker       10.0.0.3")
    print("\nSuggested demo commands inside Mininet CLI:")
    print("  h2 python3 -m http.server 80 &")
    print("  h1 curl http://10.0.0.2")
    print("  h1 ping -c 3 10.0.0.2")
    print("  h3 hping3 -S --flood -p 80 10.0.0.2")
    print("  h3 hping3 --udp --flood -p 80 10.0.0.2")
    print("\nStop attack with Ctrl+C in the Mininet CLI terminal, then run: exit\n")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    run()
