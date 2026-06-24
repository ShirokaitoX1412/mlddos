import os
import platform
import shutil
import subprocess
import threading
import time


_LAST_ALERT = {}
COOLDOWN_SECONDS = 15


def _run_non_blocking(cmd: list[str]) -> None:
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"[POPUP ERROR] {exc}")


def show_ddos_popup(event: dict) -> None:
    """
    Show popup immediately when the backend detects a DDoS attack.
    This function is called directly inside live_ips.py detection logic.
    """

    src_ip = event.get("src_ip", "unknown")
    dst_ip = event.get("dst_ip", "unknown")
    attack_type = event.get("prediction", "Unknown")
    confidence = float(event.get("confidence", 0.0))

    key = f"{src_ip}:{attack_type}"
    now = time.time()

    # Avoid showing too many popups for the same attacker
    if now - _LAST_ALERT.get(key, 0) < COOLDOWN_SECONDS:
        return

    _LAST_ALERT[key] = now

    title = "DDoS Attack Detected"
    message = (
        f"Attack type: {attack_type}\n"
        f"Source IP: {src_ip}\n"
        f"Target IP: {dst_ip}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"System action: Blocking attacker IP..."
    )

    def worker():
        os_name = platform.system().lower()

        if os_name == "linux":
            if shutil.which("notify-send"):
                _run_non_blocking([
                    "notify-send",
                    "-u", "critical",
                    "-a", "ML DDoS IPS",
                    title,
                    message,
                ])
                return

            if shutil.which("zenity"):
                _run_non_blocking([
                    "zenity",
                    "--warning",
                    f"--title={title}",
                    f"--text={message}",
                ])
                return

            print(f"\a[DDOS ALERT] {title}\n{message}")
            return

        if os_name == "windows":
            safe_title = title.replace("'", "''")
            safe_message = message.replace("'", "''")

            ps_script = (
                "Add-Type -AssemblyName PresentationFramework; "
                f"[System.Windows.MessageBox]::Show("
                f"'{safe_message}', "
                f"'{safe_title}', "
                f"'OK', "
                f"'Warning'"
                f")"
            )

            _run_non_blocking([
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script,
            ])
            return

        print(f"\a[DDOS ALERT] {title}\n{message}")

    threading.Thread(target=worker, daemon=True).start()