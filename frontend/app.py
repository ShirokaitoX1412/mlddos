"""
app.py - Professional Streamlit Dashboard for DDoS IPS

Dark-mode, high-tech UI with:
  - Analytics Tab: Model comparison, ROC curves, report figures
  - Live Monitor Tab: Real-time traffic log with RED attack alerts
  - Settings Tab: local IDS/IPS simulation controls

Usage:
    streamlit run app.py
"""

import os
from pathlib import Path
import sys
import time
from datetime import datetime

BACKEND_SRC = Path(__file__).resolve().parents[1] / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from ml_ddos.paths import AUDIT_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402

RESULTS_DIR = str(RESULTS_DIR)
MODELS_DIR = str(MODELS_DIR)
AUDIT_DIR = str(AUDIT_DIR)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "").rstrip("/")
LIVE_EVENTS_CSV = os.path.join(RESULTS_DIR, "live_events.csv")


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

    df = df.tail(limit).copy()
    for col in ("is_attack", "blocked", "simulation"):
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


def _format_metric(value, digits: int = 4) -> str:
    """Format numeric metrics consistently for the report-ready dashboard."""
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_report_ready_outputs() -> dict:
    """Load the current report-ready training outputs."""
    report_dir = Path(RESULTS_DIR)
    figures_dir = report_dir / "report_figures"
    paths = {
        "summary": report_dir / "report_ready_final_summary.csv",
        "comparison": report_dir / "report_ready_model_comparison.csv",
        "classification": report_dir / "report_ready_classification_report.csv",
        "confusion": report_dir / "report_ready_confusion_matrix.csv",
        "figures": figures_dir / "danh_muc_hinh_sinh_tu_notebook.csv",
    }
    loaded = {"paths": paths, "figures_dir": figures_dir}
    for key, path in paths.items():
        if key == "figures":
            continue
        loaded[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    loaded["figures"] = pd.read_csv(paths["figures"]) if paths["figures"].exists() else pd.DataFrame()
    return loaded


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

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="DDoS IPS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ────────────────────────────────────────
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


# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <p class="dashboard-title">DDoS IPS COMMAND CENTER</p>
    <p class="dashboard-subtitle">Real-Time Intrusion Prevention System | CICDDoS2019 | ML-Powered</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────
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

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **DDoS Detection System**
    - Dataset: CICDDoS2019
    - Models: RF, KNN, ET, MLP, XGB
    - 7 Attack Classes
    """)

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab_audit, tab3 = st.tabs([
    "📊 Analytics",
    "🔴 Live Monitor",
    "🧪 Audit",
    "⚙️ Settings",
])


# ═══════════════════════════════════════════════════════════
#  TAB 1: ANALYTICS
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Model Performance Analytics")

    outputs = _load_report_ready_outputs()
    summary_df = outputs["summary"]
    comparison_df = outputs["comparison"]
    classification_df = outputs["classification"]
    confusion_df = outputs["confusion"]
    figures_df = outputs["figures"]

    if summary_df.empty or comparison_df.empty:
        st.warning(
            "Chưa có kết quả huấn luyện mới nhất. Vui lòng chạy lại notebook báo cáo trước khi mở dashboard."
        )
    else:
        summary = summary_df.iloc[0]
        comparison_df = comparison_df.copy()
        comparison_df["Model Display"] = (
            comparison_df["Model"].astype(str)
            + " | "
            + comparison_df["Imbalance Method"].fillna("").astype(str).replace("", "baseline")
        )

        st.caption("Dashboard đang sử dụng bộ kết quả huấn luyện mới nhất của hệ thống.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            _metric_card("Best Model", str(summary.get("best_model", "N/A")), "#00d4ff")
        with col2:
            _metric_card("Accuracy", _format_metric(summary.get("accuracy")), "#55A868")
        with col3:
            _metric_card("Macro F1", _format_metric(summary.get("macro_f1")), "#C44E52")
        with col4:
            _metric_card("ROC-AUC macro OvR", _format_metric(summary.get("roc_auc_macro_ovr")), "#DD8452")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            _metric_card("Weighted F1", _format_metric(summary.get("weighted_f1")), "#64B5CD")
        with col6:
            _metric_card("False Alarm Rate", _format_metric(summary.get("false_alarm_rate")), "#ff9800")
        with col7:
            _metric_card("Attack Recall", _format_metric(summary.get("attack_recall")), "#ff1744")
        with col8:
            _metric_card("Models Compared", str(len(comparison_df)), "#b388ff")

        st.markdown("---")
        st.markdown("### Model Comparison")
        display_cols = [
            "Model",
            "Imbalance Method",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Weighted F1",
            "Minority Class Recall",
            "False Alarm Rate",
            "Attack Recall",
            "ROC AUC Macro OvR",
            "CV Macro F1 Mean",
            "CV Macro F1 Std",
            "Train/Test Gap",
            "Notes",
        ]
        existing_cols = [col for col in display_cols if col in comparison_df.columns]
        numeric_cols = comparison_df[existing_cols].select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(
            comparison_df[existing_cols].style.format({col: "{:.4f}" for col in numeric_cols}),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Best Model Detailed Evaluation")
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("#### Classification Report")
            if not classification_df.empty:
                st.dataframe(classification_df, use_container_width=True)
            else:
                st.info("Classification report CSV is not available.")

        with detail_col2:
            st.markdown("#### Confusion Matrix")
            if not confusion_df.empty:
                st.dataframe(confusion_df, use_container_width=True)
            else:
                st.info("Confusion matrix CSV is not available.")

        st.markdown("---")
        st.markdown("### Report Figures")
        if figures_df.empty:
            st.info("No generated report figure catalog found.")
        else:
            figure_options = figures_df["Tên hình"].tolist()
            default_index = 0
            selected_figure = st.selectbox("Select report figure", figure_options, index=default_index)
            figure_row = figures_df.loc[figures_df["Tên hình"] == selected_figure].iloc[0]
            figure_path = Path(figure_row["File ảnh"])
            if figure_path.exists():
                st.image(str(figure_path), caption=selected_figure, use_column_width=True)
            else:
                st.warning("Không tìm thấy hình đã chọn. Vui lòng chạy lại notebook báo cáo để sinh biểu đồ.")

            with st.expander("Show all generated report figures"):
                for _, row in figures_df.iterrows():
                    path = Path(row["File ảnh"])
                    if path.exists():
                        st.image(str(path), caption=row["Tên hình"], use_column_width=True)

        st.markdown("---")
        st.markdown("### Current Selected Model")
        selected_model_path = Path(MODELS_DIR) / "selected_model.pkl"
        if selected_model_path.exists():
            st.success("Mô hình triển khai hiện tại đã sẵn sàng.")
        else:
            st.warning("Chưa tìm thấy mô hình triển khai hiện tại. Vui lòng huấn luyện lại mô hình.")


# ═══════════════════════════════════════════════════════════
#  TAB 2: LIVE MONITOR
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Live Traffic Monitor")

    # Initialize session state
    if "ips_events" not in st.session_state:
        st.session_state.ips_events = []
    if "ips_running" not in st.session_state:
        st.session_state.ips_running = False
    if "attack_count" not in st.session_state:
        st.session_state.attack_count = 0
    if "blocked_count" not in st.session_state:
        st.session_state.blocked_count = 0

    real_events = _load_real_ips_events()
    display_events = real_events if real_events else st.session_state.ips_events
    using_real_events = bool(real_events)
    if using_real_events:
        st.success("Đang đọc sự kiện IDS/IPS thực tế từ nhật ký giám sát.")
    else:
        st.info("No real IPS CSV events yet. The demo controls below generate sample events.")

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
            ⚠️ ATTACK DETECTED: {last_attack.get('prediction', 'Unknown')}
            <br>Source: {last_attack.get('src_ip', 'N/A')} → {last_attack.get('dst_ip', 'N/A')}
            <br>Confidence: {last_attack.get('confidence', 0):.1%}
            {'<br>🔒 IP BLOCKED' if last_attack.get('blocked') else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="benign-status">
            ✓ SYSTEM NORMAL — No threats detected
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # IPS Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

    with ctrl_col1:
        if st.button("▶ Start IPS (Demo)", type="primary", use_container_width=True):
            st.session_state.ips_running = True
            if not using_real_events:
                # Only generate in-memory sample events when no replay/live log is available.
                demo_events = _generate_demo_events(threshold)
                st.session_state.ips_events.extend(demo_events)
            st.rerun()

    with ctrl_col2:
        if st.button("⏹ Stop IPS", use_container_width=True):
            st.session_state.ips_running = False
            st.rerun()

    with ctrl_col3:
        if st.button("🗑 Clear Log", use_container_width=True):
            if using_real_events:
                st.warning("Nhật ký thực tế không được xóa trên dashboard. Dừng replay và chạy lại với tùy chọn reset nếu cần làm mới dữ liệu.")
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
                icon = "🔒"
            elif is_attack:
                css_class = "log-attack"
                icon = "⚠️"
            else:
                css_class = "log-benign"
                icon = "✓"

            st.markdown(f"""
            <div class="log-entry {css_class}">
                {icon} [{event.get('timestamp', '')}]
                {event.get('src_ip', 'N/A')} → {event.get('dst_ip', 'N/A')} |
                <b>{event.get('prediction', 'Unknown')}</b>
                ({event.get('confidence', 0):.1%})
                {'| 🔒 BLOCKED' if is_blocked else ''}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No events yet. Run `python live_ips.py --list-interfaces`, then start live capture on a Windows/Npcap interface.")


# ═══════════════════════════════════════════════════════════
#  TAB 3: AUDIT
# ═══════════════════════════════════════════════════════════
with tab_audit:
    st.markdown("## Model Audit")

    summary_path = os.path.join(AUDIT_DIR, "audit_summary.md")
    metrics_path = os.path.join(AUDIT_DIR, "weighted_metrics_by_split.csv")
    per_class_path = os.path.join(AUDIT_DIR, "per_class_metrics.csv")
    gaps_path = os.path.join(AUDIT_DIR, "generalization_gaps.csv")
    dist_path = os.path.join(AUDIT_DIR, "class_distribution.csv")
    leakage_path = os.path.join(AUDIT_DIR, "split_leakage_checks.csv")
    cv_path = os.path.join(AUDIT_DIR, "cross_validation.csv")

    if not os.path.exists(metrics_path):
        st.warning("Chưa có kết quả audit mô hình. Có thể bỏ qua tab này nếu demo tập trung vào notebook báo cáo và Live Monitor.")
    else:
        st.caption("Nguồn dữ liệu: kết quả audit mô hình đã sinh trong quá trình đánh giá.")

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

        st.markdown("---")
        st.markdown("### Weighted Metrics by Split")
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df, use_container_width=True)

        if os.path.exists(gaps_path):
            st.markdown("### Generalization Gap")
            gaps_df = pd.read_csv(gaps_path)
            st.dataframe(
                gaps_df.style.highlight_max(subset=["validation_test_f1_gap"], color="#4a1010"),
                use_container_width=True,
            )

        if os.path.exists(per_class_path):
            st.markdown("### Per-Class Precision / Recall / F1")
            per_class_df = pd.read_csv(per_class_path)
            audit_models = sorted(per_class_df["model"].unique())
            audit_splits = sorted(per_class_df["split"].unique())
            selected_audit_model = st.selectbox("Audit model", audit_models, key="audit_model")
            selected_audit_split = st.selectbox("Audit split", audit_splits, index=0, key="audit_split")
            filtered_per_class = per_class_df[
                (per_class_df["model"] == selected_audit_model)
                & (per_class_df["split"] == selected_audit_split)
            ]
            st.dataframe(
                filtered_per_class.style.highlight_min(
                    subset=["precision", "recall", "f1_score"],
                    color="#4a1010",
                ),
                use_container_width=True,
            )

            safe_model = selected_audit_model.lower().replace(" ", "_")
            cm_path = os.path.join(
                AUDIT_DIR,
                f"{safe_model}_{selected_audit_split.lower()}_confusion_matrix.png",
            )
            if os.path.exists(cm_path):
                st.markdown("### Confusion Matrix")
                st.image(cm_path, use_column_width=True)

        if os.path.exists(dist_path):
            st.markdown("### Class Distribution")
            dist_df = pd.read_csv(dist_path)
            st.dataframe(dist_df, use_container_width=True)

        if os.path.exists(leakage_path):
            st.markdown("### Split / Leakage Checks")
            leakage_df = pd.read_csv(leakage_path)
            st.dataframe(leakage_df, use_container_width=True)

        if os.path.exists(cv_path):
            st.markdown("### Cross-Validation")
            cv_df = pd.read_csv(cv_path)
            st.dataframe(cv_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  TAB 4: SETTINGS
# ═══════════════════════════════════════════════════════════
with tab3:
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
        st.warning("Chưa có mô hình đã huấn luyện. Vui lòng chạy notebook báo cáo trước.")

    st.markdown("---")

    st.markdown("### IPS Configuration")
    st.code(f"""
SIMULATION_MODE = {sim_mode}
ATTACK_THRESHOLD = {threshold}
FLOW_TIMEOUT = 10.0 seconds
FLOW_CHECK_INTERVAL = 5.0 seconds
    """)

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
        st.info("Dashboard đang chạy ở chế độ cục bộ, không sử dụng Backend API bên ngoài.")

    st.markdown("---")

    st.markdown("### Quick Start Commands")
    st.code("""
# Replay IDS/IPS demo without a network card
python replay_ips.py --speed 0.2 --limit 500 --reset

# Launch Dashboard
streamlit run frontend/app.py

# Start SDN/Ryu controller on Ubuntu demo environment
./tools/run_sdn_controller.sh
    """, language="powershell")

    st.markdown("---")

    st.markdown("### System Info")
    st.json({
        "Dataset": "CICDDoS2019",
        "Task": "Multiclass DDoS detection",
        "Classes": 7,
        "Models": 5,
        "Deployment mode": "Local IDS/IPS simulation",
    })

