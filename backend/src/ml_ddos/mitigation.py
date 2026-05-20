"""
mitigation.py - OS-Level Firewall Mitigation Commands

Handles blocking/unblocking attacker IPs via:
  - Linux:   iptables -A INPUT -s <IP> -j DROP
  - Windows: netsh advfirewall firewall add rule ...

Includes a SIMULATION_MODE flag (default: True) that only prints
the command instead of executing it — safe for demos and testing.
"""

import subprocess
import platform
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mitigation")

# Safety toggle: when True, commands are printed but NOT executed
SIMULATION_MODE = True


def get_os_type() -> str:
    """Detect operating system."""
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return system


def block_ip(ip_address: str, reason: str = "DDoS Attack Detected",
             simulation: bool = None) -> dict:
    """Block an IP address using OS-level firewall rules.

    Parameters
    ----------
    ip_address : str
        IP address to block.
    reason : str
        Reason for blocking (logged).
    simulation : bool or None
        Override SIMULATION_MODE if provided.

    Returns
    -------
    dict
        Result with keys: success, command, message, timestamp, simulated
    """
    sim = simulation if simulation is not None else SIMULATION_MODE
    os_type = get_os_type()
    timestamp = datetime.now().isoformat()

    if os_type == "linux":
        command = f"iptables -A INPUT -s {ip_address} -j DROP"
    elif os_type == "windows":
        rule_name = f"BLOCK_DDoS_{ip_address.replace('.', '_')}"
        command = (
            f'netsh advfirewall firewall add rule name="{rule_name}" '
            f'dir=in action=block remoteip={ip_address} '
            f'protocol=any enable=yes'
        )
    else:
        command = f"# Unsupported OS: {os_type} — manual block required for {ip_address}"

    result = {
        "success": False,
        "command": command,
        "ip": ip_address,
        "reason": reason,
        "timestamp": timestamp,
        "simulated": sim,
        "os": os_type,
    }

    if sim:
        logger.warning(
            f"[SIMULATION] Would execute: {command} | Reason: {reason}"
        )
        result["success"] = True
        result["message"] = "SIMULATION: Command printed but NOT executed."
        return result

    # Actually execute the command
    try:
        logger.critical(f"[LIVE] Executing firewall rule: {command}")
        if os_type == "linux":
            subprocess.run(
                ["iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"],
                check=True, capture_output=True, text=True
            )
        elif os_type == "windows":
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        result["success"] = True
        result["message"] = f"IP {ip_address} blocked successfully."
        logger.critical(f"[LIVE] IP {ip_address} BLOCKED. Reason: {reason}")

    except subprocess.CalledProcessError as e:
        result["message"] = f"Failed to block IP: {e.stderr}"
        logger.error(f"[LIVE] Failed to block {ip_address}: {e.stderr}")
    except PermissionError:
        result["message"] = "Permission denied. Run as root/administrator."
        logger.error("[LIVE] Permission denied. Requires elevated privileges.")

    return result


def unblock_ip(ip_address: str, simulation: bool = None) -> dict:
    """Remove a firewall block rule for an IP address.

    Parameters
    ----------
    ip_address : str
        IP address to unblock.
    simulation : bool or None
        Override SIMULATION_MODE if provided.

    Returns
    -------
    dict
        Result dict.
    """
    sim = simulation if simulation is not None else SIMULATION_MODE
    os_type = get_os_type()
    timestamp = datetime.now().isoformat()

    if os_type == "linux":
        command = f"iptables -D INPUT -s {ip_address} -j DROP"
    elif os_type == "windows":
        rule_name = f"BLOCK_DDoS_{ip_address.replace('.', '_')}"
        command = f'netsh advfirewall firewall delete rule name="{rule_name}"'
    else:
        command = f"# Unsupported OS — manual unblock required for {ip_address}"

    result = {
        "success": False,
        "command": command,
        "ip": ip_address,
        "timestamp": timestamp,
        "simulated": sim,
        "os": os_type,
    }

    if sim:
        logger.info(f"[SIMULATION] Would execute: {command}")
        result["success"] = True
        result["message"] = "SIMULATION: Unblock command printed but NOT executed."
        return result

    try:
        logger.info(f"[LIVE] Removing firewall rule: {command}")
        if os_type == "linux":
            subprocess.run(
                ["iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"],
                check=True, capture_output=True, text=True
            )
        elif os_type == "windows":
            subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        result["success"] = True
        result["message"] = f"IP {ip_address} unblocked successfully."
        logger.info(f"[LIVE] IP {ip_address} unblocked.")

    except subprocess.CalledProcessError as e:
        result["message"] = f"Failed to unblock IP: {e.stderr}"
        logger.error(f"[LIVE] Failed to unblock {ip_address}: {e.stderr}")

    return result


def list_blocked_ips(simulation: bool = None) -> list:
    """List currently blocked IPs (Linux only via iptables)."""
    sim = simulation if simulation is not None else SIMULATION_MODE
    os_type = get_os_type()

    if os_type != "linux":
        logger.info(f"[{os_type}] Listing blocked IPs not supported on this OS.")
        return []

    if sim:
        logger.info("[SIMULATION] Would run: iptables -L INPUT -n --line-numbers")
        return []

    try:
        result = subprocess.run(
            ["iptables", "-L", "INPUT", "-n", "--line-numbers"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        blocked = []
        for line in lines:
            if "DROP" in line:
                parts = line.split()
                for part in parts:
                    if _is_ip(part):
                        blocked.append(part)
                        break
        return blocked
    except Exception as e:
        logger.error(f"Failed to list blocked IPs: {e}")
        return []


def _is_ip(s: str) -> bool:
    """Check if a string looks like an IPv4 address."""
    parts = s.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


if __name__ == "__main__":
    print(f"OS: {get_os_type()}")
    print(f"SIMULATION_MODE: {SIMULATION_MODE}")
    print()

    # Demo: simulate blocking
    r = block_ip("192.168.1.100", reason="Demo: TCP SYN Flood detected")
    print(f"Result: {r}")
    print()

    r2 = unblock_ip("192.168.1.100")
    print(f"Result: {r2}")
