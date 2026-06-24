"""One-click launcher for the DDoS IDS/IPS dashboard.

Double-click this file on the Victim machine, or run:
    python app.py

The launcher starts Streamlit with ``frontend/app.py``. When the file is run
from Streamlit directly, it falls back to loading the frontend app in-process.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import runpy
import shlex
import shutil
import socket
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_APP = PROJECT_ROOT / "frontend" / "app.py"
LIVE_IPS_APP = PROJECT_ROOT / "live_ips.py"
RESULTS_DIR = PROJECT_ROOT / "results"
LIVE_EVENTS_CSV = RESULTS_DIR / "live_events.csv"
TASK_NAME = "MLDDoS Victim Agent"
SHORTCUT_NAME = "DDoS IPS Dashboard.lnk"
LINUX_SHORTCUT_NAME = "DDoS IPS Dashboard.desktop"
LINUX_SERVICE_NAME = "mlddos-victim-agent.service"
LINUX_SERVICE_PATH = Path("/etc/systemd/system") / LINUX_SERVICE_NAME
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8501
DASHBOARD_URL = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
DASHBOARD_LAUNCHER_LOG = RESULTS_DIR / "dashboard_launcher.log"


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _launch_streamlit() -> int:
    if platform.system().lower() == "linux":
        return _launch_kali_app_window()

    command = [
        _python_path(),
        "-m",
        "streamlit",
        "run",
        str(FRONTEND_APP),
        "--server.headless=false",
    ]
    print("Starting DDoS IDS/IPS dashboard...")
    print("Press Ctrl+C in this window to stop the dashboard.")
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_dashboard(timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_port_open(DASHBOARD_HOST, DASHBOARD_PORT):
            return True
        time.sleep(0.3)
    return False


def _app_browser_command(url: str) -> list[str] | None:
    chromium_like = [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "brave-browser",
        "microsoft-edge",
    ]
    for name in chromium_like:
        binary = shutil.which(name)
        if binary:
            return [
                binary,
                f"--app={url}",
                "--new-window",
                "--no-first-run",
                "--disable-translate",
            ]

    for name in ["firefox-esr", "firefox"]:
        binary = shutil.which(name)
        if binary:
            return [binary, "--new-window", url]

    xdg_open = shutil.which("xdg-open")
    if xdg_open:
        return [xdg_open, url]
    return None


def _launch_kali_app_window() -> int:
    """Run Streamlit locally and open it as a standalone app-like window."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    streamlit_process = None

    with open(DASHBOARD_LAUNCHER_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launch dashboard\n")
        log_file.flush()

        if not _is_port_open(DASHBOARD_HOST, DASHBOARD_PORT):
            command = [
                _python_path(),
                "-m",
                "streamlit",
                "run",
                str(FRONTEND_APP),
                "--server.headless=true",
                "--server.address",
                DASHBOARD_HOST,
                "--server.port",
                str(DASHBOARD_PORT),
                "--browser.gatherUsageStats=false",
            ]
            streamlit_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not _wait_for_dashboard():
                log_file.write("ERROR: Streamlit did not start in time.\n")
                return 1

        browser_command = _app_browser_command(DASHBOARD_URL)
        if not browser_command:
            log_file.write("ERROR: No browser found to open dashboard.\n")
            return 1

        log_file.write("Opening dashboard window: " + " ".join(browser_command) + "\n")
        log_file.flush()
        subprocess.Popen(browser_command, stdout=log_file, stderr=subprocess.STDOUT)

    if streamlit_process is not None:
        return streamlit_process.wait()
    return 0


def _python_path(gui: bool = False) -> str:
    """Return the best Python executable for this project."""
    if platform.system().lower().startswith("win"):
        venv_dir = PROJECT_ROOT / "venv" / "Scripts"
        candidate = venv_dir / ("pythonw.exe" if gui else "python.exe")
        if candidate.exists():
            return str(candidate)
        if gui:
            candidate = Path(sys.executable).with_name("pythonw.exe")
            if candidate.exists():
                return str(candidate)
        return sys.executable

    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return shutil.which("python3") or sys.executable


def _pythonw_path() -> str:
    """Prefer pythonw.exe on Windows so launcher/agent can run like an app."""
    return _python_path(gui=True)


def _desktop_dir() -> Path:
    if platform.system().lower().startswith("win"):
        return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        home = Path("/home") / sudo_user
    else:
        home = Path.home()
    localized = home / "Màn hình"
    if localized.exists():
        return localized
    desktop = home / "Desktop"
    if desktop.exists():
        return desktop
    desktop.mkdir(parents=True, exist_ok=True)
    return desktop


def _chown_to_login_user(path: Path) -> None:
    sudo_user = os.environ.get("SUDO_USER")
    if not sudo_user or sudo_user == "root":
        return
    try:
        shutil.chown(path, user=sudo_user, group=sudo_user)
    except Exception:
        pass


def _run_powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )


def _create_desktop_shortcut() -> Path:
    """Create a Windows desktop shortcut that opens the dashboard."""
    shortcut_path = _desktop_dir() / SHORTCUT_NAME
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    target = _pythonw_path()
    arguments = f'"{PROJECT_ROOT / "app.py"}"'
    icon = target
    ps_script = f"""
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut('{shortcut_path}')
$shortcut.TargetPath = '{target}'
$shortcut.Arguments = '{arguments}'
$shortcut.WorkingDirectory = '{PROJECT_ROOT}'
$shortcut.IconLocation = '{icon},0'
$shortcut.Description = 'Open DDoS IDS/IPS Dashboard'
$shortcut.Save()
"""
    result = _run_powershell(ps_script)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return shortcut_path


def _create_linux_desktop_shortcut() -> Path:
    """Create a Linux/Kali desktop launcher that opens the dashboard."""
    shortcut_path = _desktop_dir() / LINUX_SHORTCUT_NAME
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python_bin = _python_path()
    app_path = PROJECT_ROOT / "app.py"
    shell_command = (
        f"cd {shlex.quote(str(PROJECT_ROOT))} && "
        f"{shlex.quote(python_bin)} {shlex.quote(str(app_path))} --app-window "
        f">> {shlex.quote(str(DASHBOARD_LAUNCHER_LOG))} 2>&1"
    )
    exec_command = f"sh -lc {shlex.quote(shell_command)}"
    content = f"""[Desktop Entry]
Type=Application
Name=DDoS IPS Dashboard
Comment=Open DDoS IDS/IPS Dashboard
Exec={exec_command}
Path={PROJECT_ROOT}
Terminal=false
Icon=utilities-system-monitor
Categories=Network;Security;Education;
"""
    shortcut_path.write_text(content, encoding="utf-8")
    shortcut_path.chmod(shortcut_path.stat().st_mode | 0o755)
    _chown_to_login_user(shortcut_path)
    gio = shutil.which("gio")
    if gio:
        subprocess.run(
            [gio, "set", str(shortcut_path), "metadata::trusted", "true"],
            capture_output=True,
            text=True,
        )
    return shortcut_path


def _agent_task_command(
    *,
    model: str,
    interface: str,
    victim_ip: str,
    threshold: float,
    live_agent: bool,
    mitigation_backend: str,
    cicflowmeter: bool,
    cicflowmeter_cmd: str,
    cic_window: float,
) -> str:
    args = [
        f'"{_pythonw_path()}"',
        f'"{LIVE_IPS_APP}"',
        "--model",
        model,
        "--threshold",
        str(threshold),
        "--events-csv",
        f'"{LIVE_EVENTS_CSV}"',
        "--mitigation-backend",
        mitigation_backend,
    ]
    if interface:
        args.extend(["--interface", f'"{interface}"'])
    if victim_ip:
        args.extend(["--victim-ip", f'"{victim_ip}"'])
    if live_agent:
        args.append("--live")
    if cicflowmeter:
        args.append("--cicflowmeter")
        args.extend(["--cic-window", str(cic_window)])
        if cic_window <= 2:
            args.append("--fast-log")
        if cicflowmeter_cmd:
            args.extend(["--cicflowmeter-cmd", f'"{cicflowmeter_cmd}"'])
    return " ".join(args)


def _create_startup_task(
    *,
    model: str,
    interface: str,
    victim_ip: str,
    threshold: float,
    live_agent: bool,
    mitigation_backend: str,
    cicflowmeter: bool,
    cicflowmeter_cmd: str,
    cic_window: float,
) -> bool:
    """Create a Windows Scheduled Task that starts the agent on user logon."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_command = _agent_task_command(
        model=model,
        interface=interface,
        victim_ip=victim_ip,
        threshold=threshold,
        live_agent=live_agent,
        mitigation_backend=mitigation_backend,
        cicflowmeter=cicflowmeter,
        cicflowmeter_cmd=cicflowmeter_cmd,
        cic_window=cic_window,
    )
    base = [
        "schtasks",
        "/Create",
        "/TN",
        TASK_NAME,
        "/SC",
        "ONLOGON",
        "/TR",
        task_command,
        "/F",
    ]
    elevated = base + ["/RL", "HIGHEST"]
    result = subprocess.run(elevated, capture_output=True, text=True)
    if result.returncode == 0:
        return True

    # Fallback for non-admin installs. The agent can still detect and show
    # popups, but live firewall blocking will need Administrator privileges.
    fallback = subprocess.run(base, capture_output=True, text=True)
    if fallback.returncode != 0:
        raise RuntimeError(fallback.stderr.strip() or result.stderr.strip())
    return False


def _start_agent_task() -> None:
    subprocess.run(
        ["schtasks", "/Run", "/TN", TASK_NAME],
        capture_output=True,
        text=True,
    )


def _delete_startup_task() -> None:
    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
    )


def _linux_agent_command(
    *,
    model: str,
    interface: str,
    victim_ip: str,
    threshold: float,
    live_agent: bool,
    mitigation_backend: str,
    cicflowmeter: bool,
    cicflowmeter_cmd: str,
    cic_window: float,
) -> str:
    python_bin = _python_path()
    args = [
        python_bin,
        str(LIVE_IPS_APP),
        "--model",
        model,
        "--threshold",
        str(threshold),
        "--events-csv",
        str(LIVE_EVENTS_CSV),
        "--mitigation-backend",
        mitigation_backend,
    ]
    if interface:
        args.extend(["--interface", interface])
    if victim_ip:
        args.extend(["--victim-ip", victim_ip])
    if live_agent:
        args.append("--live")
    if cicflowmeter:
        args.append("--cicflowmeter")
        args.extend(["--cic-window", str(cic_window)])
        if cic_window <= 2:
            args.append("--fast-log")
        if cicflowmeter_cmd:
            args.extend(["--cicflowmeter-cmd", cicflowmeter_cmd])
    return " ".join(shlex.quote(part) for part in args)


def _create_linux_systemd_service(
    *,
    model: str,
    interface: str,
    victim_ip: str,
    threshold: float,
    live_agent: bool,
    mitigation_backend: str,
    cicflowmeter: bool,
    cicflowmeter_cmd: str,
    cic_window: float,
) -> None:
    if os.geteuid() != 0:
        raise PermissionError("Linux/Kali startup agent requires sudo/root privileges.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    command = _linux_agent_command(
        model=model,
        interface=interface,
        victim_ip=victim_ip,
        threshold=threshold,
        live_agent=live_agent,
        mitigation_backend=mitigation_backend,
        cicflowmeter=cicflowmeter,
        cicflowmeter_cmd=cicflowmeter_cmd,
        cic_window=cic_window,
    )
    py_path = PROJECT_ROOT / "backend" / "src"
    service = f"""[Unit]
Description=ML DDoS Victim IDS/IPS Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={PROJECT_ROOT}
Environment=PYTHONPATH={py_path}
ExecStart={command}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
    LINUX_SERVICE_PATH.write_text(service, encoding="utf-8")
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", LINUX_SERVICE_NAME], check=True)


def _delete_linux_systemd_service() -> None:
    if os.geteuid() != 0:
        print("WARNING: Linux/Kali service removal requires sudo/root privileges.")
        return
    subprocess.run(["systemctl", "disable", "--now", LINUX_SERVICE_NAME], capture_output=True, text=True)
    if LINUX_SERVICE_PATH.exists():
        LINUX_SERVICE_PATH.unlink()
    subprocess.run(["systemctl", "daemon-reload"], capture_output=True, text=True)


def install_windows_victim_app(args: argparse.Namespace) -> int:
    if not platform.system().lower().startswith("win"):
        print("Victim app installer is currently intended for Windows VM only.")
        return 2

    shortcut_path = _create_desktop_shortcut()
    highest = _create_startup_task(
        model=args.model,
        interface=args.interface or "",
        victim_ip=args.victim_ip or "",
        threshold=args.threshold,
        live_agent=args.live_agent,
        mitigation_backend=args.mitigation_backend,
        cicflowmeter=args.cicflowmeter,
        cicflowmeter_cmd=args.cicflowmeter_cmd,
        cic_window=args.cic_window,
    )
    _start_agent_task()

    print("DDoS IDS/IPS Victim app installed.")
    print(f"Desktop shortcut: {shortcut_path}")
    print(f"Startup task: {TASK_NAME}")
    print(f"Agent mode: {'LIVE BLOCKING' if args.live_agent else 'SIMULATION'}")
    print(f"Mitigation backend: {args.mitigation_backend}")
    print(f"Run level: {'highest privileges' if highest else 'current user'}")
    if args.live_agent and not highest:
        print("WARNING: Live blocking may fail unless the task runs as Administrator.")
    return 0


def install_linux_victim_app(args: argparse.Namespace) -> int:
    if platform.system().lower() != "linux":
        print("Linux/Kali installer can only run on Linux.")
        return 2

    shortcut_path = _create_linux_desktop_shortcut()
    _create_linux_systemd_service(
        model=args.model,
        interface=args.interface or "",
        victim_ip=args.victim_ip or "",
        threshold=args.threshold,
        live_agent=args.live_agent,
        mitigation_backend=args.mitigation_backend,
        cicflowmeter=args.cicflowmeter,
        cicflowmeter_cmd=args.cicflowmeter_cmd,
        cic_window=args.cic_window,
    )

    print("DDoS IDS/IPS Victim app installed.")
    print(f"Desktop shortcut: {shortcut_path}")
    print(f"Startup service: {LINUX_SERVICE_NAME}")
    print(f"Agent mode: {'LIVE BLOCKING' if args.live_agent else 'SIMULATION'}")
    print(f"Mitigation backend: {args.mitigation_backend}")
    return 0


def uninstall_windows_victim_app() -> int:
    _delete_startup_task()
    shortcut_path = _desktop_dir() / SHORTCUT_NAME
    if shortcut_path.exists():
        shortcut_path.unlink()
    print("DDoS IDS/IPS Victim app removed.")
    return 0


def uninstall_linux_victim_app() -> int:
    _delete_linux_systemd_service()
    shortcut_path = _desktop_dir() / LINUX_SHORTCUT_NAME
    if shortcut_path.exists():
        shortcut_path.unlink()
    print("DDoS IDS/IPS Victim app removed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDoS IDS/IPS Dashboard launcher")
    parser.add_argument("--install-victim-app", action="store_true",
                        help="Install Victim app shortcut and startup agent on Windows/Linux")
    parser.add_argument("--uninstall-victim-app", action="store_true",
                        help="Remove Victim app shortcut and startup agent on Windows/Linux")
    parser.add_argument("--interface", default="",
                        help="Capture interface for the startup agent")
    parser.add_argument("--victim-ip", default="",
                        help="Victim IP filter for the startup agent")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Attack confidence threshold for the startup agent")
    parser.add_argument("--model", default="selected_model",
                        help="Saved model name for the startup agent")
    parser.add_argument("--mitigation-backend", default="auto",
                        choices=["auto", "iptables", "nftables", "windows_firewall", "manual"],
                        help="Firewall backend for the startup agent")
    parser.add_argument("--live-agent", action="store_true",
                        help="Enable real firewall blocking for the startup agent")
    parser.add_argument("--cicflowmeter", action="store_true",
                        help="Run the startup agent with CICFlowMeter feature extraction")
    parser.add_argument("--cicflowmeter-cmd", default=os.getenv("CICFLOWMETER_CMD", ""),
                        help="CICFlowMeter command template for the startup agent")
    parser.add_argument("--cic-window", type=float, default=2.0,
                        help="CICFlowMeter capture window in seconds (default: 2 for faster demo logs)")
    parser.add_argument("--app-window", action="store_true",
                        help="Open dashboard in an app-like browser window on Kali/Linux")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.app_window:
        return _launch_kali_app_window()
    if args.install_victim_app:
        system = platform.system().lower()
        if system.startswith("win"):
            return install_windows_victim_app(args)
        if system == "linux":
            return install_linux_victim_app(args)
        print(f"Unsupported victim app OS: {platform.system()}")
        return 2
    if args.uninstall_victim_app:
        system = platform.system().lower()
        if system.startswith("win"):
            return uninstall_windows_victim_app()
        if system == "linux":
            return uninstall_linux_victim_app()
        print(f"Unsupported victim app OS: {platform.system()}")
        return 2

    if _running_under_streamlit():
        runpy.run_path(str(FRONTEND_APP), run_name="__main__")
        return 0

    return _launch_streamlit()


if __name__ == "__main__":
    raise SystemExit(main())
