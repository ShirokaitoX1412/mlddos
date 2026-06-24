"""
OS-level mitigation helpers for the DDoS IDS/IPS demo.

The victim machine can be Linux or Windows:
- Linux: uses iptables when available, otherwise nftables when available.
- Windows: uses Windows Firewall through netsh advfirewall.

By default, mitigation runs in simulation mode and only prints the command.
Use live mode only in a controlled lab and with root/administrator privileges.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mitigation")

SIMULATION_MODE = True


def get_os_type() -> str:
    """Return normalized OS name."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "linux":
        return "linux"
    if system == "darwin":
        return "macos"
    return system or "unknown"


def is_elevated() -> bool:
    """Return whether current process can modify firewall rules."""
    os_type = get_os_type()
    if os_type == "windows":
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    if os_type == "linux":
        return hasattr(os, "geteuid") and os.geteuid() == 0
    return False


def get_mitigation_backend(preferred: str = "auto") -> str:
    """Choose the firewall backend for the current OS."""
    preferred = (preferred or "auto").lower()
    if preferred != "auto":
        return preferred

    os_type = get_os_type()
    if os_type == "windows":
        return "windows_firewall"
    if os_type == "linux":
        if shutil.which("iptables"):
            return "iptables"
        if shutil.which("nft"):
            return "nftables"
        return "linux_manual"
    return "manual"


def get_mitigation_status(preferred_backend: str = "auto") -> dict[str, Any]:
    """Return OS/backend information for dashboard and logs."""
    backend = get_mitigation_backend(preferred_backend)
    return {
        "os": get_os_type(),
        "backend": backend,
        "elevated": is_elevated(),
        "simulation_default": SIMULATION_MODE,
        "iptables_available": bool(shutil.which("iptables")),
        "nft_available": bool(shutil.which("nft")),
        "netsh_available": bool(shutil.which("netsh")) if get_os_type() == "windows" else False,
    }


def _rule_name(ip_address: str) -> str:
    safe_ip = ip_address.replace(".", "_").replace(":", "_")
    return f"MLDDoS_BLOCK_{safe_ip}"


def _build_block_command(
    ip_address: str,
    backend: str,
) -> tuple[str, list[str] | str | None, bool]:
    rule_name = _rule_name(ip_address)

    if backend == "iptables":
        argv = ["iptables", "-I", "INPUT", "-s", ip_address, "-j", "DROP"]
        return " ".join(argv), argv, False

    if backend == "nftables":
        argv = ["nft", "add", "rule", "inet", "filter", "input", "ip", "saddr", ip_address, "drop"]
        return " ".join(argv), argv, False

    if backend == "windows_firewall":
        command = (
            f'netsh advfirewall firewall add rule name="{rule_name}" '
            f'dir=in action=block remoteip={ip_address} protocol=any profile=any enable=yes'
        )
        return command, command, True

    return f"# Unsupported backend {backend}: manually block {ip_address}", None, False


def _build_unblock_command(
    ip_address: str,
    backend: str,
) -> tuple[str, list[str] | str | None, bool]:
    rule_name = _rule_name(ip_address)

    if backend == "iptables":
        argv = ["iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"]
        return " ".join(argv), argv, False

    if backend == "nftables":
        command = "# nftables delete requires a rule handle; inspect with: nft list ruleset"
        return command, None, False

    if backend == "windows_firewall":
        command = f'netsh advfirewall firewall delete rule name="{rule_name}"'
        return command, command, True

    return f"# Unsupported backend {backend}: manually unblock {ip_address}", None, False


def _base_result(ip_address: str, reason: str, simulation: bool, backend: str, command: str) -> dict[str, Any]:
    return {
        "success": False,
        "command": command,
        "ip": ip_address,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "simulated": simulation,
        "os": get_os_type(),
        "backend": backend,
        "elevated": is_elevated(),
    }


def block_ip(
    ip_address: str,
    reason: str = "DDoS Attack Detected",
    simulation: bool | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Block an attacker IP with the correct OS firewall command."""
    sim = simulation if simulation is not None else SIMULATION_MODE
    selected_backend = get_mitigation_backend(backend)
    command, argv, use_shell = _build_block_command(ip_address, selected_backend)
    result = _base_result(ip_address, reason, sim, selected_backend, command)

    if sim:
        logger.warning("[SIMULATION] Would execute: %s | Reason: %s", command, reason)
        result["success"] = True
        result["message"] = "SIMULATION: firewall command was not executed."
        return result

    if argv is None:
        result["message"] = f"Unsupported mitigation backend: {selected_backend}"
        logger.error(result["message"])
        return result

    if not result["elevated"]:
        result["message"] = "Permission denied. Run as root/administrator."
        logger.error("[LIVE] Permission denied. Requires elevated privileges.")
        return result

    try:
        logger.critical("[LIVE] Executing firewall rule: %s", command)
        completed = subprocess.run(
            argv,
            shell=use_shell,
            check=True,
            capture_output=True,
            text=True,
        )
        result["success"] = True
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["message"] = f"IP {ip_address} blocked successfully."
        logger.critical("[LIVE] IP %s BLOCKED. Reason: %s", ip_address, reason)
    except subprocess.CalledProcessError as exc:
        result["stdout"] = exc.stdout
        result["stderr"] = exc.stderr
        result["message"] = f"Failed to block IP: {exc.stderr or exc}"
        logger.error("[LIVE] Failed to block %s: %s", ip_address, result["message"])
    except FileNotFoundError as exc:
        result["message"] = f"Firewall command not found: {exc}"
        logger.error(result["message"])

    return result


def unblock_ip(
    ip_address: str,
    simulation: bool | None = None,
    backend: str = "auto",
) -> dict[str, Any]:
    """Remove a previously installed firewall block when supported."""
    sim = simulation if simulation is not None else SIMULATION_MODE
    selected_backend = get_mitigation_backend(backend)
    command, argv, use_shell = _build_unblock_command(ip_address, selected_backend)
    result = _base_result(ip_address, "Unblock requested", sim, selected_backend, command)

    if sim:
        logger.info("[SIMULATION] Would execute: %s", command)
        result["success"] = True
        result["message"] = "SIMULATION: unblock command was not executed."
        return result

    if argv is None:
        result["message"] = f"Unsupported unblock operation for backend: {selected_backend}"
        logger.error(result["message"])
        return result

    if not result["elevated"]:
        result["message"] = "Permission denied. Run as root/administrator."
        logger.error("[LIVE] Permission denied. Requires elevated privileges.")
        return result

    try:
        logger.info("[LIVE] Removing firewall rule: %s", command)
        completed = subprocess.run(
            argv,
            shell=use_shell,
            check=True,
            capture_output=True,
            text=True,
        )
        result["success"] = True
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["message"] = f"IP {ip_address} unblocked successfully."
    except subprocess.CalledProcessError as exc:
        result["stdout"] = exc.stdout
        result["stderr"] = exc.stderr
        result["message"] = f"Failed to unblock IP: {exc.stderr or exc}"
        logger.error("[LIVE] Failed to unblock %s: %s", ip_address, result["message"])

    return result


def list_blocked_ips(simulation: bool | None = None, backend: str = "auto") -> list[str]:
    """List blocked IPs for iptables. Other backends return an empty list."""
    sim = simulation if simulation is not None else SIMULATION_MODE
    selected_backend = get_mitigation_backend(backend)

    if selected_backend != "iptables":
        logger.info("[%s] Listing blocked IPs is not implemented.", selected_backend)
        return []

    if sim:
        logger.info("[SIMULATION] Would run: iptables -L INPUT -n --line-numbers")
        return []

    try:
        result = subprocess.run(
            ["iptables", "-L", "INPUT", "-n", "--line-numbers"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        logger.error("Failed to list blocked IPs: %s", exc)
        return []

    blocked = []
    for line in result.stdout.splitlines():
        if "DROP" not in line:
            continue
        for part in line.split():
            if _is_ip(part):
                blocked.append(part)
                break
    return blocked


def _is_ip(value: str) -> bool:
    parts = value.split(".")
    if len(parts) != 4:
        return False
    return all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)


if __name__ == "__main__":
    print(get_mitigation_status())
    print(block_ip("192.168.56.10", reason="Demo attack", simulation=True))
    print(unblock_ip("192.168.56.10", simulation=True))
