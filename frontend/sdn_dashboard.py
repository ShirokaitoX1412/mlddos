"""
Streamlit dashboard for the SDN/Ryu DDoS demo.

This UI visualizes the SDN experiment while Ryu and Mininet run in terminals.
It reads results/live_events.csv, which is written by sdn_ryu_detector.py.

Run:
    streamlit run frontend/sdn_dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st


BACKEND_SRC = Path(__file__).resolve().parents[1] / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from ml_ddos.paths import RESULTS_DIR  # noqa: E402


LIVE_EVENTS_CSV = RESULTS_DIR / "live_events.csv"


st.set_page_config(
    page_title="SDN/Ryu DDoS IDS Demo",
    page_icon="🛡️",
    layout="wide",
)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@st.cache_data(ttl=1)
def load_events(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for col in ("is_attack", "blocked", "simulation"):
        if col in df.columns:
            df[col] = df[col].map(as_bool)
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def topology_dot() -> str:
    return """
digraph {
  graph [rankdir=LR, bgcolor="transparent", pad="0.2"];
  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, margin="0.16,0.10"];
  edge [color="#64748b", penwidth=1.6, fontname="Arial", fontsize=10];

  h1 [label="h1\\nBenign Client\\n10.0.0.1", fillcolor="#dcfce7", color="#16a34a"];
  h2 [label="h2\\nVictim Server\\n10.0.0.2", fillcolor="#e0f2fe", color="#0284c7"];
  h3 [label="h3\\nAttacker\\n10.0.0.3", fillcolor="#fee2e2", color="#dc2626"];
  s1 [label="s1\\nOpen vSwitch\\nOpenFlow 1.3", fillcolor="#f8fafc", color="#475569"];
  ryu [label="Ryu Controller\\nML IDS/IPS", fillcolor="#fef3c7", color="#d97706"];
  model [label="selected_model.pkl\\nDDoS Classifier", fillcolor="#ede9fe", color="#7c3aed"];
  log [label="results/live_events.csv\\nDashboard Log", fillcolor="#f1f5f9", color="#475569"];

  h1 -> s1 [label="normal traffic"];
  h3 -> s1 [label="flood traffic", color="#dc2626"];
  s1 -> h2 [label="forward/drop"];
  s1 -> ryu [label="flow stats"];
  ryu -> s1 [label="OpenFlow rules"];
  ryu -> model [label="predict"];
  ryu -> log [label="write events"];
}
"""


def command_block(title: str, command: str) -> None:
    st.markdown(f"**{title}**")
    st.code(command, language="bash")


def render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    attacks = int(df["is_attack"].sum()) if "is_attack" in df else 0
    blocked = int(df["blocked"].sum()) if "blocked" in df else 0
    benign = total - attacks
    latest = "No events"
    if not df.empty and "prediction" in df.columns:
        latest = str(df.iloc[-1]["prediction"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", total)
    c2.metric("Benign", benign)
    c3.metric("Attack Alerts", attacks)
    c4.metric("Blocked Flows", blocked, help="Only increases when MLDDOS_OBSERVE_ONLY=0")
    st.caption(f"Latest prediction: {latest}")


def render_charts(df: pd.DataFrame) -> None:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Attack Type Distribution")
        if not df.empty and "prediction" in df.columns:
            counts = df["prediction"].fillna("Unknown").value_counts()
            st.bar_chart(counts)
        else:
            st.info("No SDN events yet.")

    with right:
        st.subheader("Confidence Timeline")
        if not df.empty and {"timestamp_dt", "confidence"}.issubset(df.columns):
            chart_df = df.dropna(subset=["timestamp_dt"]).tail(100).set_index("timestamp_dt")
            if not chart_df.empty:
                st.line_chart(chart_df[["confidence"]])
            else:
                st.info("Waiting for timestamped events.")
        else:
            st.info("Waiting for confidence data.")


def render_event_table(df: pd.DataFrame) -> None:
    st.subheader("Live SDN Events")
    if df.empty:
        st.info(f"Waiting for events at: {LIVE_EVENTS_CSV}")
        return

    columns = [
        "timestamp", "src_ip", "dst_ip", "protocol", "prediction",
        "confidence", "is_attack", "blocked", "simulation", "interface",
    ]
    columns = [col for col in columns if col in df.columns]
    display_df = df.tail(200)[columns].iloc[::-1]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_commands() -> None:
    st.subheader("Demo Commands")
    tabs = st.tabs(["Controller", "Topology", "Traffic", "Blocking"])

    with tabs[0]:
        command_block(
            "Run Ryu controller in observe-only mode",
            "source ryu-venv/bin/activate\n./tools/run_sdn_controller.sh",
        )
    with tabs[1]:
        command_block(
            "Run Mininet topology",
            "sudo python3 tools/mininet_sdn_topology.py",
        )
    with tabs[2]:
        command_block(
            "Normal traffic inside Mininet CLI",
            "h2 python3 -m http.server 80 &\nh1 curl http://10.0.0.2\nh1 ping -c 3 10.0.0.2",
        )
        command_block(
            "Attack traffic inside Mininet CLI",
            "h3 hping3 -S --flood -p 80 10.0.0.2\nh3 hping3 --udp --flood -p 80 10.0.0.2",
        )
    with tabs[3]:
        command_block(
            "Enable OpenFlow blocking",
            "MLDDOS_OBSERVE_ONLY=0 MLDDOS_THRESHOLD=0.95 ./tools/run_sdn_controller.sh",
        )
        command_block(
            "Inspect Open vSwitch rules",
            "sudo ovs-ofctl -O OpenFlow13 dump-flows s1",
        )


def main() -> None:
    st.title("SDN/Ryu DDoS IDS/IPS Demo")
    st.caption("Visual dashboard for Mininet + Open vSwitch + Ryu Controller + ML model")

    with st.sidebar:
        st.header("Runtime")
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh interval", 1, 10, 2)
        st.write("Events CSV")
        st.code(str(LIVE_EVENTS_CSV), language="text")
        st.write("Current mode is controlled by:")
        st.code("MLDDOS_OBSERVE_ONLY=1/0", language="bash")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.cache_data.clear()
        st.rerun()

    df = load_events(str(LIVE_EVENTS_CSV))

    st.subheader("Topology")
    st.graphviz_chart(topology_dot(), use_container_width=True)

    render_metrics(df)
    render_charts(df)
    render_event_table(df)
    render_commands()

    st.markdown("---")
    st.caption(
        "Ryu and Mininet still run in terminal because they control Linux networking. "
        "This UI is the presentation layer for topology, alerts, and mitigation evidence."
    )


if __name__ == "__main__":
    main()
