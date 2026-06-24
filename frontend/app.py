"""
app.py - Streamlit dashboard for the DDoS IDS/IPS demo.

The dashboard focuses on the live two-VM demo:
  - Live Monitor: real-time IDS/IPS events.
  - Settings: local agent and attack command reference.
"""

import os
import ipaddress
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from ml_ddos.paths import MODELS_DIR, RESULTS_DIR  # noqa: E402

RESULTS_DIR = str(RESULTS_DIR)
MODELS_DIR = str(MODELS_DIR)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "").rstrip("/")
LIVE_EVENTS_CSV = os.path.join(RESULTS_DIR, "live_events.csv")
LIVE_IPS_PROCESS_LOG = os.path.join(RESULTS_DIR, "live_ips_process.log")
LIVE_IPS_ENTRYPOINT = PROJECT_ROOT / "live_ips.py"
LINUX_SERVICE_NAME = "mlddos-victim-agent.service"
DEFAULT_DISPLAY_ATTACK_THRESHOLD = 0.80
LIVE_EVENT_COLUMNS = [
    "timestamp", "src_ip", "dst_ip", "protocol",
    "fwd_packets", "bwd_packets", "prediction", "confidence",
    "is_attack", "blocked", "simulation", "model", "interface",
    "victim_os", "mitigation_backend", "mitigation_elevated",
    "block_command", "block_message", "source",
]


def _get_backend_health() -> dict:
    """Check deployed backend API health when BACKEND_API_URL is configured."""
    if not BACKEND_API_URL:
        return {"configured": False, "healthy": False, "message": "BACKEND_API_URL is not configured"}

    try:
        import requests

        resp = requests.get(f"{BACKEND_API_URL}/health", timeout=5)
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        return {
            "configured": True,
            "healthy": resp.ok,
            "status_code": resp.status_code,
            "response": body,
            "url": BACKEND_API_URL,
        }
    except Exception as e:
        return {
            "configured": True,
            "healthy": False,
            "error": str(e),
            "url": BACKEND_API_URL,
        }


def _is_process_running(process) -> bool:
    """Return True when a background subprocess is still alive."""
    return process is not None and process.poll() is None


def _start_live_ips_process(
    interface: str,
    victim_ip: str,
    threshold: float,
    simulation: bool,
    mitigation_backend: str,
):
    """Start live_ips.py from the dashboard without adding another script."""
    base_command = [
        sys.executable,
        str(LIVE_IPS_ENTRYPOINT),
        "--model",
        "selected_model",
        "--threshold",
        str(threshold),
        "--events-csv",
        LIVE_EVENTS_CSV,
        "--mitigation-backend",
        mitigation_backend,
        "--cicflowmeter",
        "--cic-window",
        "2",
        "--fast-log",
        "--cicflowmeter-cmd",
        "cicflowmeter -f {pcap} -c {csv}",
    ]
    if interface.strip():
        base_command.extend(["--interface", interface.strip()])
    if victim_ip.strip():
        base_command.extend(["--victim-ip", victim_ip.strip()])
    if not simulation:
        base_command.append("--live")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_SRC)
    command = base_command
    process_env = env

    if hasattr(os, "geteuid") and os.geteuid() != 0:
        pkexec = shutil.which("pkexec")
        if not pkexec:
            raise RuntimeError("Root permission is required for packet capture. Install pkexec or run dashboard with sudo.")
        command = [pkexec, "env", f"PYTHONPATH={BACKEND_SRC}", *base_command]
        process_env = None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LIVE_IPS_PROCESS_LOG, "w", encoding="utf-8") as log_file:
        log_file.write("Command: " + " ".join(str(part) for part in command) + "\n")

    log_file = open(LIVE_IPS_PROCESS_LOG, "a", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=process_env,
        stdout=log_file,
        stderr=log_file,
        text=True,
    )


def _stop_live_ips_process():
    """Stop the dashboard-managed live IPS process if it exists."""
    process = st.session_state.get("live_ips_process")
    if _is_process_running(process):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    st.session_state.live_ips_process = None


def _reset_live_events_log() -> None:
    """Start a clean live monitoring log for the current demo run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(columns=LIVE_EVENT_COLUMNS).to_csv(LIVE_EVENTS_CSV, index=False)


def _tail_file(path: str, lines: int = 12) -> str:
    if not os.path.exists(path):
        return "No process log was created."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            content = file.readlines()
        return "".join(content[-lines:]).strip() or "Process log is empty."
    except Exception as exc:
        return f"Could not read process log: {exc}"


def _systemd_service_status(service_name: str = LINUX_SERVICE_NAME) -> dict:
    """Read Kali/Linux background agent status."""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return {"available": False, "active": False, "state": "unavailable", "detail": "systemctl not found"}

    state = subprocess.run(
        [systemctl, "is-active", service_name],
        capture_output=True,
        text=True,
    )
    enabled = subprocess.run(
        [systemctl, "is-enabled", service_name],
        capture_output=True,
        text=True,
    )
    return {
        "available": True,
        "active": state.stdout.strip() == "active",
        "state": state.stdout.strip() or state.stderr.strip() or "unknown",
        "enabled": enabled.stdout.strip() or enabled.stderr.strip() or "unknown",
    }


def _systemd_recent_logs(service_name: str = LINUX_SERVICE_NAME, lines: int = 25) -> str:
    journalctl = shutil.which("journalctl")
    if not journalctl:
        return "journalctl not found"
    result = subprocess.run(
        [journalctl, "-u", service_name, "-n", str(lines), "--no-pager"],
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr or "").strip()


def _generate_demo_events(threshold: float, n: int = 15) -> list:
    """Generate demo IPS events for dashboard testing."""
    np.random.seed(int(time.time()) % 1000)
    attack_types = [
        "Benign", "TCP SYN Flood", "UDP Flood", "LDAP Flood",
        "MSSQL Flood", "NetBIOS Flood", "UDP-Lag Flood"
    ]
    events = []

    for _ in range(n):
        is_attack = np.random.random() < 0.4
        if is_attack:
            pred = np.random.choice(attack_types[1:])
            conf = np.random.uniform(0.85, 0.99)
        else:
            pred = "Benign"
            conf = np.random.uniform(0.92, 0.99)

        blocked = is_attack and conf > threshold

        events.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "src_ip": f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
            "dst_ip": f"10.0.0.{np.random.randint(1,20)}",
            "protocol": np.random.choice([6, 17]),
            "fwd_packets": np.random.randint(1, 5000),
            "bwd_packets": np.random.randint(0, 100),
            "prediction": pred,
            "confidence": conf,
            "is_attack": is_attack,
            "blocked": blocked,
        })
    return events


def _as_bool(value) -> bool:
    """Normalize CSV booleans written by live_ips.py."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_noise_destination(ip_value) -> bool:
    """Hide broadcast/multicast background traffic from attack alert panels."""
    try:
        ip_obj = ipaddress.ip_address(str(ip_value).strip())
    except ValueError:
        return False
    return ip_obj.is_multicast or str(ip_obj).endswith(".255")


def _is_confirmed_attack(event: dict, threshold: float) -> bool:
    """Treat an event as an attack only when confidence is high enough."""
    if not event.get("is_attack", False):
        return False
    if _is_noise_destination(event.get("dst_ip", "")):
        return False
    return float(event.get("confidence", 0.0) or 0.0) >= threshold


def _load_real_ips_events(limit: int = 500) -> list:
    """Load real live IPS events written to results/live_events.csv."""
    if not os.path.exists(LIVE_EVENTS_CSV):
        return []

    try:
        df = pd.read_csv(LIVE_EVENTS_CSV)
    except Exception:
        return []

    if df.empty:
        return []

    if "source" in df.columns:
        df = df[df["source"].fillna("").astype(str).str.lower().isin(["live_ips", ""])]
    elif "actual_label" in df.columns:
        # Ignore old offline-demo logs on the real-time Victim dashboard.
        return []

    if df.empty:
        return []

    df = df.tail(limit).copy()
    for col in ("is_attack", "blocked", "simulation", "mitigation_elevated"):
        if col in df.columns:
            df[col] = df[col].map(_as_bool)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    for col in ("fwd_packets", "bwd_packets", "protocol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df.to_dict("records")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column available in a metrics CSV."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _metric_column(df: pd.DataFrame, metric: str) -> str | None:
    """Support both legacy metrics and robust CV/test metric column names."""
    aliases = {
        "accuracy": ["Accuracy", "CV Accuracy Mean", "Test Accuracy"],
        "precision": ["Precision", "CV Precision Mean", "Test Precision"],
        "recall": ["Recall", "CV Recall Mean", "Test Recall"],
        "f1": ["F1-Score", "CV F1-Score Mean", "Test F1-Score"],
        "f1_std": ["CV F1-Score Std"],
        "f1_macro": ["F1-Macro", "CV F1-Macro Mean", "Test F1-Macro"],
        "roc_auc": ["ROC-AUC", "Test ROC-AUC", "CV ROC-AUC Mean"],
    }
    return _first_existing_column(df, aliases.get(metric, []))


def _highlight_metric_columns(df: pd.DataFrame) -> list[str]:
    """Select existing metric columns for Streamlit table highlighting."""
    cols = []
    for metric in ("accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"):
        col = _metric_column(df, metric)
        if col:
            cols.append(col)
    return cols


def _metric_card(label: str, value: str, color: str = "#00d4ff") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}; font-size:1.7em">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# â”€â”€â”€ Page Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="DDoS IPS Dashboard",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€ Dark Theme CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    .metric-value {
        font-size: 2.2em;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }

    .metric-label {
        font-size: 0.85em;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Alert box */
    .attack-alert {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4em;
        font-weight: 700;
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.4);
        margin: 10px 0;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 30px rgba(255, 0, 0, 0.4); }
        50% { opacity: 0.85; box-shadow: 0 0 50px rgba(255, 0, 0, 0.7); }
    }

    .benign-status {
        background: linear-gradient(135deg, #00c853, #009624);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2em;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(0, 200, 83, 0.3);
    }

    /* Table styling */
    .dataframe {
        background: #161b22 !important;
        color: #e0e0e0 !important;
    }

    /* Header */
    .dashboard-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid #0f3460;
        margin-bottom: 20px;
    }

    .dashboard-title {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .dashboard-subtitle {
        color: #8b949e;
        font-size: 1em;
        margin-top: 5px;
    }

    /* Log entry */
    .log-entry {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
        border-left: 4px solid;
    }

    .log-benign {
        background: rgba(0, 200, 83, 0.1);
        border-left-color: #00c853;
        color: #a5d6a7;
    }

    .log-attack {
        background: rgba(255, 0, 0, 0.15);
        border-left-color: #ff1744;
        color: #ff8a80;
    }

    .log-blocked {
        background: rgba(255, 152, 0, 0.15);
        border-left-color: #ff9800;
        color: #ffcc80;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e0e0e0;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        border-color: #00d4ff;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


# â”€â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">DDoS IPS COMMAND CENTER</p>
    <p class="dashboard-subtitle">Kali Victim Live IDS/IPS Monitor</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    service_status = _systemd_service_status()

    st.markdown("### IPS Agent")
    st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    if service_status["active"]:
        st.success("Agent service: ACTIVE")
    else:
        st.error(f"Agent service: {service_status['state']}")
    st.caption(f"Startup: {service_status.get('enabled', 'unknown')}")
    st.caption("Victim interface: eth1 / 192.168.56.103")

    st.markdown("---")
    st.markdown("### Alert Filter")
    display_attack_threshold = st.slider(
        "Minimum attack confidence",
        0.50,
        1.00,
        DEFAULT_DISPLAY_ATTACK_THRESHOLD,
        0.05,
    )
    st.caption("Only high-confidence ML detections are shown as attacks.")

    st.markdown("---")
    st.markdown("### Install / Repair")
    st.code(
        "sudo python3 app.py --install-victim-app --interface eth1 --live-agent",
        language="bash",
    )
    st.caption("Ubuntu Attacker sends traffic. Kali Victim agent detects and blocks automatically.")


st.markdown("## Live IDS/IPS Monitor")

if "ips_events" not in st.session_state:
    st.session_state.ips_events = []
if "live_ips_process" not in st.session_state:
    st.session_state.live_ips_process = None
if "live_ips_message" not in st.session_state:
    st.session_state.live_ips_message = ""

st.session_state.ips_running = service_status["active"]

real_events = _load_real_ips_events()
display_events = real_events if real_events else st.session_state.ips_events
visible_events = [
    event for event in display_events
    if not _is_noise_destination(event.get("dst_ip", ""))
]
confirmed_attacks = [
    event for event in visible_events
    if _is_confirmed_attack(event, display_attack_threshold)
]
using_real_events = bool(real_events)

if using_real_events:
    st.success("Reading live IDS/IPS events from the monitoring log.")
elif service_status["active"]:
    st.info("Agent is running. Waiting for traffic from Ubuntu Attacker.")
else:
    st.warning("Agent service is not running. Install or repair the service from the sidebar command.")

if st.session_state.live_ips_message:
    st.caption(st.session_state.live_ips_message)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    _metric_card("Total Events", str(len(visible_events)))
with status_col2:
    attacks = len(confirmed_attacks)
    _metric_card("Attacks Detected", str(attacks), "#ff1744")
with status_col3:
    blocked = sum(1 for event in visible_events if event.get("blocked"))
    _metric_card("IPs Blocked", str(blocked), "#ff9800")
with status_col4:
    ips_status = "ACTIVE" if st.session_state.ips_running else "STANDBY"
    status_color = "#00c853" if st.session_state.ips_running else "#8b949e"
    _metric_card("IPS Status", ips_status, status_color)

st.markdown("---")

recent_attacks = [
    event for event in visible_events[-30:]
    if _is_confirmed_attack(event, display_attack_threshold)
]
if recent_attacks:
    last_attack = recent_attacks[-1]
    st.markdown(f"""
    <div class="attack-alert">
        ATTACK DETECTED: {last_attack.get('prediction', 'Unknown')}
        <br>Source: {last_attack.get('src_ip', 'N/A')} -> {last_attack.get('dst_ip', 'N/A')}
        <br>Confidence: {last_attack.get('confidence', 0):.1%}
        {'<br>IP BLOCKED' if last_attack.get('blocked') else ''}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="benign-status">
        SYSTEM NORMAL - No threats detected
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

ctrl_col1, ctrl_col2 = st.columns([1, 2])
with ctrl_col1:
    if st.button("Refresh Log", type="primary", use_container_width=True):
        st.rerun()
with ctrl_col2:
    st.caption("The IPS agent runs automatically as a Kali system service, similar to a firewall.")

if not service_status["active"]:
    with st.expander("Service diagnostic log"):
        st.code(_systemd_recent_logs(), language="text")

st.markdown("### Monitoring Log")
if display_events:
    for event in reversed(visible_events[-80:]):
        is_attack = _is_confirmed_attack(event, display_attack_threshold)
        is_blocked = event.get("blocked", False)

        if is_blocked:
            css_class = "log-blocked"
            icon = "[BLOCKED]"
        elif is_attack:
            css_class = "log-attack"
            icon = "[ATTACK]"
        elif event.get("is_attack", False):
            css_class = "log-benign"
            icon = "[LOW CONF]"
        else:
            css_class = "log-benign"
            icon = "[OK]"

        st.markdown(f"""
        <div class="log-entry {css_class}">
            {icon} [{event.get('timestamp', '')}]
            {event.get('src_ip', 'N/A')} -> {event.get('dst_ip', 'N/A')} |
            <b>{event.get('prediction', 'Unknown')}</b>
            ({event.get('confidence', 0):.1%})
            | {event.get('victim_os', 'kali')}/{event.get('mitigation_backend', 'auto')}
            {'| BLOCKED' if is_blocked else ''}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No monitoring log yet.")

st.stop()


# â”€â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("### System Status")
    st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    backend_health = _get_backend_health()
    if backend_health["configured"]:
        if backend_health["healthy"]:
            st.success("Backend API: Online")
        else:
            st.error("Backend API: Offline")
        with st.expander("Backend API"):
            st.code(BACKEND_API_URL)
            st.json(backend_health)
    else:
        st.info("Backend API: Local mode")

    st.markdown("---")
    st.markdown("### IPS Settings")
    sim_mode = st.toggle("Simulation Mode", value=True)
    threshold = st.slider("Attack Threshold", 0.5, 1.0, 0.95, 0.01)
    capture_interface = st.text_input(
        "Capture Interface",
        value="",
        placeholder="Leave empty for auto, or enter eth0/enp0s3...",
        help="On Kali Victim, check the interface name with: ip a",
    )
    victim_ip = st.text_input(
        "Victim IP",
        value="192.168.56.103",
        help="Only flows whose destination is this IP are evaluated.",
    )
    mitigation_backend = st.selectbox(
        "Mitigation Backend",
        ["auto", "iptables", "nftables", "windows_firewall", "manual"],
        index=0,
        help="auto selects iptables/nftables on Kali/Linux.",
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **DDoS Detection System**
    - Dataset: CICDDoS2019
    - Models: RF, KNN, ET, MLP, XGB
    - 7 Attack Classes
    """)

# â”€â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab_live, tab_settings = st.tabs(["Live Monitor", "Settings"])


# Live Monitor
with tab_live:
    st.markdown("## Live Traffic Monitor")

    # Initialize session state
    if "ips_events" not in st.session_state:
        st.session_state.ips_events = []
    if "live_ips_process" not in st.session_state:
        st.session_state.live_ips_process = None
    if "live_ips_message" not in st.session_state:
        st.session_state.live_ips_message = ""
    st.session_state.ips_running = _is_process_running(st.session_state.live_ips_process)
    if "attack_count" not in st.session_state:
        st.session_state.attack_count = 0
    if "blocked_count" not in st.session_state:
        st.session_state.blocked_count = 0

    real_events = _load_real_ips_events()
    display_events = real_events if real_events else st.session_state.ips_events
    using_real_events = bool(real_events)
    if using_real_events:
        st.success("Reading real IDS/IPS events from the monitoring log.")
    else:
        st.info("No real event yet. Click Start IPS on the Kali Victim, then run traffic from Ubuntu Attacker.")
    if st.session_state.live_ips_message:
        st.caption(st.session_state.live_ips_message)

    # Status cards
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-value">{len(display_events)}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        attacks = sum(1 for e in display_events if e.get("is_attack"))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attacks Detected</div>
            <div class="metric-value" style="color: #ff1744">{attacks}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        blocked = sum(1 for e in display_events if e.get("blocked"))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IPs Blocked</div>
            <div class="metric-value" style="color: #ff9800">{blocked}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_col4:
        ips_status = "ACTIVE" if st.session_state.ips_running else "STANDBY"
        status_color = "#00c853" if st.session_state.ips_running else "#8b949e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">IPS Status</div>
            <div class="metric-value" style="color: {status_color}; font-size:1.4em">{ips_status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Check for recent attacks
    recent_attacks = [e for e in display_events[-10:] if e.get("is_attack")]
    if recent_attacks:
        last_attack = recent_attacks[-1]
        st.markdown(f"""
        <div class="attack-alert">
            ATTACK DETECTED: {last_attack.get('prediction', 'Unknown')}
            <br>Source: {last_attack.get('src_ip', 'N/A')} -> {last_attack.get('dst_ip', 'N/A')}
            <br>Confidence: {last_attack.get('confidence', 0):.1%}
            {'<br>IP BLOCKED' if last_attack.get('blocked') else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="benign-status">
            SYSTEM NORMAL - No threats detected
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # IPS Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        if st.button("Start IPS", type="primary", use_container_width=True):
            if _is_process_running(st.session_state.live_ips_process):
                st.session_state.live_ips_message = "IPS is already running."
            else:
                try:
                    st.session_state.live_ips_process = _start_live_ips_process(
                        capture_interface,
                        victim_ip,
                        threshold,
                        sim_mode,
                        mitigation_backend,
                    )
                    mode = "simulation" if sim_mode else "live blocking"
                    iface = capture_interface.strip() or "auto"
                    st.session_state.live_ips_message = (
                        f"Started live_ips.py ({mode}) on interface: {iface}, "
                        f"firewall backend: {mitigation_backend}. "
                        "If no event appears, check capture permission and interface name."
                    )
                except Exception as exc:
                    st.session_state.live_ips_message = f"Could not start IPS: {exc}"
            st.rerun()

    with ctrl_col2:
        if st.button("Stop IPS", use_container_width=True):
            _stop_live_ips_process()
            st.session_state.live_ips_message = "Stopped live_ips.py."
            st.rerun()

    with ctrl_col3:
        if st.button("Clear Log", use_container_width=True):
            if using_real_events:
                st.warning("Real monitoring logs are not cleared from the dashboard. Stop IPS and reset the log manually if needed.")
            else:
                st.session_state.ips_events = []
                st.rerun()

    # Event log
    st.markdown("### Event Log")
    if display_events:
        for event in reversed(display_events[-50:]):
            is_attack = event.get("is_attack", False)
            is_blocked = event.get("blocked", False)

            if is_blocked:
                css_class = "log-blocked"
                icon = "[BLOCKED]"
            elif is_attack:
                css_class = "log-attack"
                icon = "[ATTACK]"
            else:
                css_class = "log-benign"
                icon = "[OK]"

            st.markdown(f"""
            <div class="log-entry {css_class}">
                {icon} [{event.get('timestamp', '')}]
                {event.get('src_ip', 'N/A')} -> {event.get('dst_ip', 'N/A')} |
                <b>{event.get('prediction', 'Unknown')}</b>
                ({event.get('confidence', 0):.1%})
                | {event.get('victim_os', 'os?')}/{event.get('mitigation_backend', 'backend?')}
                {'| BLOCKED' if is_blocked else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No event yet. On Kali Victim click Start IPS, then run traffic commands from Ubuntu Attacker.")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Settings
with tab_settings:
    st.markdown("## System Configuration")

    st.markdown("### Model Selection")
    model_files = []
    if os.path.exists(MODELS_DIR):
        model_files = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

    if model_files:
        preferred = "selected_model"
        default_index = model_files.index(preferred) if preferred in model_files else 0
        selected = st.selectbox("Active Model for IPS", model_files, index=default_index)
        st.info(f"Selected model: **{selected}** - used for live traffic classification")
    else:
        st.warning("No trained model found. Please train the model first.")

    st.markdown("---")

    st.markdown("### IPS Configuration")
    st.code(f"""
SIMULATION_MODE = {sim_mode}
ATTACK_THRESHOLD = {threshold}
FLOW_TIMEOUT = 10.0 seconds
FLOW_CHECK_INTERVAL = 5.0 seconds
    """)

    st.markdown("---")
    st.markdown("### Victim App Install")
    st.code("""
# Kali Victim - run once with sudo
sudo python3 app.py --install-victim-app --interface eth0 --live-agent

# Kali Victim safe demo mode without real firewall blocking
sudo python3 app.py --install-victim-app --interface eth0

# Remove desktop shortcut and startup agent on Kali
sudo python3 app.py --uninstall-victim-app
    """, language="bash")
    st.info(
        "Sau khi cài, Kali Victim sẽ có icon DDoS IPS Dashboard ngoài Desktop. "
        "Agent tự chạy nền khi Kali khởi động, còn icon dùng để mở dashboard như một cửa sổ ứng dụng."
    )

    st.markdown("---")
    st.markdown("### Backend API")
    if BACKEND_API_URL:
        if st.button("Test Backend API", use_container_width=True):
            health = _get_backend_health()
            if health["healthy"]:
                st.success("Backend API is reachable.")
            else:
                st.error("Backend API is not reachable.")
            st.json(health)
    else:
        st.info("Dashboard is running in local mode without an external Backend API.")

    st.markdown("---")
    st.markdown("### Ubuntu Attacker Commands")
    st.code("""
# Run on Ubuntu Attacker
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack benign --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack syn_flood --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack udp_flood --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack mixed --duration 60 --pps 600 --spoof-sources 10.10.1.10,10.10.1.11,10.10.1.12,10.10.1.13,10.10.1.14
    """, language="bash")
    st.info(
        "Ubuntu Attacker dùng lệnh để sinh từng loại lưu lượng kiểm thử. "
        "Kali Victim chỉ cần chạy agent và mở dashboard để quan sát phát hiện, cảnh báo và chặn IP. "
        "Lệnh mixed mô phỏng nhiều nguồn tấn công cùng lúc để phục vụ phần demo bảo vệ."
    )

    st.markdown("---")

    st.markdown("### Quick Start Commands")
    st.code("""
# Launch Dashboard
streamlit run frontend/app.py

# Ubuntu Attacker traffic demo
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack benign --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack syn_flood --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack udp_flood --duration 20
sudo python3 tools/ddos_traffic_generator.py --target <victim-ip> --attack mixed --duration 60 --pps 600 --spoof-sources 10.10.1.10,10.10.1.11,10.10.1.12,10.10.1.13,10.10.1.14
    """, language="powershell")

    st.markdown("---")

    st.markdown("### System Info")
    st.json({
        "Dataset": "CICDDoS2019",
        "Task": "Multiclass DDoS detection",
        "Classes": 7,
        "Models": 5,
        "Deployment mode": "Local IDS/IPS demo",
    })

