"""
app.py - Professional Streamlit Dashboard for DDoS IPS

Dark-mode, high-tech UI with:
  - Analytics Tab: Model comparison, ROC curves, SHAP explanations
  - Live Monitor Tab: Real-time traffic log with RED attack alerts
  - Settings Tab: Telegram integration for instant attack alerts

Usage:
    streamlit run app.py
"""

import os
import time
import json
import threading
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

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

    # Telegram settings
    st.markdown("---")
    st.markdown("### Telegram Alerts")
    telegram_token = st.text_input("Bot Token", type="password", key="tg_token")
    telegram_chat_id = st.text_input("Chat ID", key="tg_chat")
    if st.button("Test Telegram"):
        if telegram_token and telegram_chat_id:
            _send_telegram(telegram_token, telegram_chat_id,
                           "DDoS IPS Dashboard: Test alert!")
            st.success("Test message sent!")
        else:
            st.error("Enter both Token and Chat ID")

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


def _send_telegram(token: str, chat_id: str, message: str):
    """Send a Telegram message."""
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=5)
    except Exception as e:
        st.error(f"Telegram error: {e}")


# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🔴 Live Monitor", "⚙️ Settings"])


# ═══════════════════════════════════════════════════════════
#  TAB 1: ANALYTICS
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Model Performance Analytics")

    # Load scores
    val_csv = os.path.join(RESULTS_DIR, "validation_scores.csv")
    test_csv = os.path.join(RESULTS_DIR, "test_scores.csv")

    if os.path.exists(val_csv) and os.path.exists(test_csv):
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        # Top metrics cards
        col1, col2, col3, col4 = st.columns(4)
        best_model = val_df.loc[val_df["F1-Score"].idxmax()]

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Model</div>
                <div class="metric-value" style="font-size:1.4em">{best_model['Model']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{best_model['Accuracy']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{best_model['F1-Score']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Models Trained</div>
                <div class="metric-value">5</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Comparison tables
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Validation Results")
            st.dataframe(
                val_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                                           color="#0f3460"),
                use_container_width=True
            )

        with col_right:
            st.markdown("### Test Results")
            st.dataframe(
                test_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                                            color="#0f3460"),
                use_container_width=True
            )

        st.markdown("---")

        # Model comparison chart
        comp_img = os.path.join(RESULTS_DIR, "model_comparison.png")
        if os.path.exists(comp_img):
            st.markdown("### Model Comparison")
            st.image(comp_img, use_container_width=True)

        st.markdown("---")

        # Visualizations grid
        st.markdown("### Per-Model Visualizations")
        model_names = ["random_forest", "knn", "extra_trees", "mlp_classifier", "xgboost"]
        display_names = ["Random Forest", "KNN", "Extra Trees", "MLP Classifier", "XGBoost"]

        selected_model = st.selectbox("Select Model", display_names)
        safe_model = selected_model.lower().replace(" ", "_")

        viz_col1, viz_col2 = st.columns(2)

        cm_path = os.path.join(RESULTS_DIR, f"{safe_model}_confusion_matrix.png")
        roc_path = os.path.join(RESULTS_DIR, f"{safe_model}_roc_curve.png")
        fi_path = os.path.join(RESULTS_DIR, f"{safe_model}_feature_importance.png")
        shap_path = os.path.join(RESULTS_DIR, f"{safe_model}_shap_summary_bar.png")

        with viz_col1:
            if os.path.exists(cm_path):
                st.markdown("#### Confusion Matrix")
                st.image(cm_path, use_container_width=True)
            if os.path.exists(fi_path):
                st.markdown("#### Feature Importance")
                st.image(fi_path, use_container_width=True)

        with viz_col2:
            if os.path.exists(roc_path):
                st.markdown("#### ROC Curve")
                st.image(roc_path, use_container_width=True)
            if os.path.exists(shap_path):
                st.markdown("#### SHAP Explanation")
                st.image(shap_path, use_container_width=True)

    else:
        st.warning("No results found. Run `python main.py` first to train models.")


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

    # Status cards
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Events</div>
            <div class="metric-value">{len(st.session_state.ips_events)}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        attacks = sum(1 for e in st.session_state.ips_events if e.get("is_attack"))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attacks Detected</div>
            <div class="metric-value" style="color: #ff1744">{attacks}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        blocked = sum(1 for e in st.session_state.ips_events if e.get("blocked"))
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
    recent_attacks = [e for e in st.session_state.ips_events[-10:] if e.get("is_attack")]
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
            # Generate demo events
            demo_events = _generate_demo_events()
            st.session_state.ips_events.extend(demo_events)
            st.rerun()

    with ctrl_col2:
        if st.button("⏹ Stop IPS", use_container_width=True):
            st.session_state.ips_running = False
            st.rerun()

    with ctrl_col3:
        if st.button("🗑 Clear Log", use_container_width=True):
            st.session_state.ips_events = []
            st.rerun()

    # Event log
    st.markdown("### Event Log")
    if st.session_state.ips_events:
        for event in reversed(st.session_state.ips_events[-50:]):
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
        st.info("No events yet. Start the IPS or run `sudo python live_ips.py` for real traffic.")

    # Telegram alert for attacks
    if recent_attacks and telegram_token and telegram_chat_id:
        for atk in recent_attacks:
            msg = (
                f"🚨 DDoS ALERT!\n"
                f"Type: {atk['prediction']}\n"
                f"Source: {atk['src_ip']}\n"
                f"Confidence: {atk['confidence']:.1%}\n"
                f"Blocked: {'Yes' if atk['blocked'] else 'No'}"
            )
            _send_telegram(telegram_token, telegram_chat_id, msg)


def _generate_demo_events(n: int = 15) -> list:
    """Generate demo IPS events for dashboard testing."""
    np.random.seed(int(time.time()) % 1000)
    attack_types = [
        "Benign", "TCP SYN Flood", "UDP Flood", "LDAP Flood",
        "MSSQL Flood", "NetBIOS Flood", "UDP-Lag Flood"
    ]
    events = []

    for i in range(n):
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


# ═══════════════════════════════════════════════════════════
#  TAB 3: SETTINGS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## System Configuration")

    st.markdown("### Model Selection")
    model_files = []
    if os.path.exists(MODELS_DIR):
        model_files = [f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

    if model_files:
        selected = st.selectbox("Active Model for IPS", model_files, index=0)
        st.info(f"Selected model: **{selected}** — used for live traffic classification")
    else:
        st.warning("No trained models found. Run `python main.py` first.")

    st.markdown("---")

    st.markdown("### IPS Configuration")
    st.code(f"""
SIMULATION_MODE = {sim_mode}
ATTACK_THRESHOLD = {threshold}
FLOW_TIMEOUT = 10.0 seconds
FLOW_CHECK_INTERVAL = 5.0 seconds
    """)

    st.markdown("---")

    st.markdown("### Telegram Alert Setup")
    st.markdown("""
    **How to set up Telegram alerts:**
    1. Create a bot via [@BotFather](https://t.me/BotFather)
    2. Copy the **Bot Token** to the sidebar
    3. Send a message to your bot, then get your **Chat ID** from
       `https://api.telegram.org/bot<TOKEN>/getUpdates`
    4. Enter the Chat ID in the sidebar
    5. Click **Test Telegram** to verify
    """)

    st.markdown("---")

    st.markdown("### Quick Start Commands")
    st.code("""
# Train models & generate reports
python main.py

# Generate SHAP explanations
python shap_explainer.py

# Start IPS (simulation mode)
sudo python live_ips.py

# Start IPS (LIVE mode — actually blocks IPs)
sudo python live_ips.py --live

# Launch Dashboard
streamlit run app.py
    """, language="bash")

    st.markdown("---")

    st.markdown("### System Info")
    st.json({
        "Dataset": "CICDDoS2019",
        "Features (raw)": 78,
        "Features (processed)": 32,
        "Classes": 7,
        "Models": 5,
        "OS": os.name,
    })
