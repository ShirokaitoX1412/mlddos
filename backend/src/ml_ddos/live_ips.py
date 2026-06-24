"""
live_ips.py - Real-Time Intrusion Prevention System (IPS) Engine

Uses Scapy to sniff live network traffic, aggregates packets into flows,
and classifies them using a trained ML model. If attack probability > 95%,
triggers mitigation (firewall block).

SIMULATION_MODE (default: True) — prints block commands without executing.

Usage:
    python live_ips.py --list-interfaces
    python live_ips.py --interface "Npcap Loopback Adapter"
    python live_ips.py --interface "<interface-name>" --live  # DANGEROUS
"""

import os
import sys
import time
import pickle
import argparse
import csv
import importlib
import logging
import shlex
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

try:
    from scapy.interfaces import get_if_list
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.sendrecv import sniff
    from scapy.utils import wrpcap
except ImportError:
    print("[live_ips] Scapy not installed. Install: pip install scapy")
    sys.exit(1)

from .mitigation import block_ip, get_mitigation_status
from .alert_notifier import show_ddos_popup
from .paths import DATA_DIR as PROJECT_DATA_DIR
from .paths import MODELS_DIR as PROJECT_MODELS_DIR
from .paths import RESULTS_DIR as PROJECT_RESULTS_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("live_ips")

# Paths
MODELS_DIR = str(PROJECT_MODELS_DIR)
RESULTS_DIR = str(PROJECT_RESULTS_DIR)
DATA_DIR = str(PROJECT_DATA_DIR)
DEFAULT_EVENTS_CSV = os.path.join(RESULTS_DIR, "live_events.csv")

# IPS Configuration
ATTACK_THRESHOLD = 0.95       # Probability threshold for blocking
FLOW_TIMEOUT = 10.0           # Seconds before a flow is considered complete
FLOW_CHECK_INTERVAL = 5.0     # Seconds between flow aggregation checks
DEFAULT_MODEL = "selected_model"


CICDDOS_FEATURE_COLUMNS = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Avg Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init Fwd Win Bytes", "Init Bwd Win Bytes",
    "Fwd Act Data Packets", "Fwd Seg Size Min", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
    "Idle Min",
]


CICFLOWMETER_TO_CICDDOS_COLUMNS = {
    "protocol": "Protocol",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Fwd Packets Length Total",
    "totlen_bwd_pkts": "Bwd Packets Length Total",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Packet Length Min",
    "pkt_len_max": "Packet Length Max",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwr_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Avg Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init Fwd Win Bytes",
    "init_bwd_win_byts": "Init Bwd Win Bytes",
    "fwd_act_data_pkts": "Fwd Act Data Packets",
    "fwd_seg_size_min": "Fwd Seg Size Min",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}


CICFLOW_TIME_COLUMNS_US = [
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

DEPLOYMENT_LABELS = {
    "Benign",
    "LDAP Flood",
    "MSSQL Flood",
    "NetBIOS Flood",
    "TCP SYN Flood",
    "UDP Flood",
    "UDP-Lag Flood",
}


class FlowAggregator:
    """Aggregates raw packets into network flows (5-tuple keyed).

    A flow is defined by: (src_ip, dst_ip, src_port, dst_port, protocol)
    """

    def __init__(self):
        self.flows = defaultdict(lambda: {
            "packets": [],
            "start_time": None,
            "last_time": None,
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "protocol": 0,
            "flags": defaultdict(int),
            "fwd_pkt_lengths": [],
            "bwd_pkt_lengths": [],
            "flow_iat": [],
            "fwd_iat": [],
            "bwd_iat": [],
            "fwd_header_bytes": 0,
            "bwd_header_bytes": 0,
            "fwd_seg_sizes": [],
            "fwd_data_packets": 0,
            "init_fwd_win_bytes": -1,
            "init_bwd_win_bytes": -1,
            "src_ip": "",
            "dst_ip": "",
            "src_port": 0,
            "dst_port": 0,
            "dst_ports": defaultdict(int),
        })
        self.lock = threading.Lock()

    def add_packet(self, packet):
        """Process a captured packet and add to the appropriate flow."""
        if not packet.haslayer(IP):
            return

        ip = packet[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto = ip.proto
        src_port = 0
        dst_port = 0
        pkt_len = len(ip)
        header_len = int(getattr(ip, "ihl", 0) or 0) * 4
        payload_len = 0
        tcp_window = -1

        if packet.haslayer(TCP):
            tcp = packet[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            header_len += int(getattr(tcp, "dataofs", 0) or 0) * 4
            payload_len = len(tcp.payload)
            tcp_window = int(getattr(tcp, "window", -1))
        elif packet.haslayer(UDP):
            udp = packet[UDP]
            src_port = udp.sport
            dst_port = udp.dport
            header_len += 8
            payload_len = len(udp.payload)

        flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
        rev_key = (dst_ip, src_ip, dst_port, src_port, proto)
        now = time.time()

        with self.lock:
            # Determine flow direction
            if flow_key in self.flows:
                fkey = flow_key
                is_fwd = True
            elif rev_key in self.flows:
                fkey = rev_key
                is_fwd = False
            else:
                fkey = flow_key
                is_fwd = True

            flow = self.flows[fkey]

            if flow["start_time"] is None:
                flow["start_time"] = now
                flow["src_ip"] = src_ip
                flow["dst_ip"] = dst_ip
                flow["protocol"] = proto
                flow["src_port"] = src_port
                flow["dst_port"] = dst_port

            if dst_port:
                flow["dst_ports"][int(dst_port)] += 1

            # IAT calculations
            if flow["last_time"] is not None:
                iat = now - flow["last_time"]
                flow["flow_iat"].append(iat)
                if is_fwd:
                    flow["fwd_iat"].append(iat)
                else:
                    flow["bwd_iat"].append(iat)

            flow["last_time"] = now

            if is_fwd:
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += pkt_len
                flow["fwd_pkt_lengths"].append(pkt_len)
                flow["fwd_header_bytes"] += header_len
                flow["fwd_seg_sizes"].append(header_len)
                if payload_len > 0:
                    flow["fwd_data_packets"] += 1
                if flow["init_fwd_win_bytes"] < 0 and tcp_window >= 0:
                    flow["init_fwd_win_bytes"] = tcp_window
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += pkt_len
                flow["bwd_pkt_lengths"].append(pkt_len)
                flow["bwd_header_bytes"] += header_len
                if flow["init_bwd_win_bytes"] < 0 and tcp_window >= 0:
                    flow["init_bwd_win_bytes"] = tcp_window

            # Track TCP flags
            if packet.haslayer(TCP):
                tcp_flags = packet[TCP].flags
                if tcp_flags & 0x01:
                    flow["flags"]["FIN"] += 1
                if tcp_flags & 0x02:
                    flow["flags"]["SYN"] += 1
                if tcp_flags & 0x04:
                    flow["flags"]["RST"] += 1
                if tcp_flags & 0x08:
                    flow["flags"]["PSH"] += 1
                if tcp_flags & 0x10:
                    flow["flags"]["ACK"] += 1
                if tcp_flags & 0x20:
                    flow["flags"]["URG"] += 1
                if tcp_flags & 0x40:
                    flow["flags"]["ECE"] += 1
                if tcp_flags & 0x80:
                    flow["flags"]["CWE"] += 1

    def get_completed_flows(self, timeout: float = FLOW_TIMEOUT):
        """Return idle flows or sufficiently large active flows for ML scoring."""
        now = time.time()
        completed = {}

        with self.lock:
            expired_keys = []
            for key, flow in self.flows.items():
                if not flow["last_time"]:
                    continue

                idle_expired = (now - flow["last_time"]) > timeout
                duration = (
                    flow["last_time"] - flow["start_time"]
                    if flow["start_time"] and flow["last_time"]
                    else 0
                )
                active_large_flow = (
                    duration >= 3.0
                    and (flow.get("fwd_packets", 0) + flow.get("bwd_packets", 0)) >= 50
                )

                if idle_expired or active_large_flow:
                    completed[key] = flow.copy()
                    expired_keys.append(key)

            for key in expired_keys:
                del self.flows[key]

        return completed


def flow_to_features(flow: dict, n_features: int = 32) -> np.ndarray:
    """Convert a flow dictionary into a feature vector matching the trained model.

    Extracts the 32 features used by the preprocessed CICDDoS2019 model.
    Feature order must match the training pipeline.
    """
    fwd_lengths = flow["fwd_pkt_lengths"] or [0]
    bwd_lengths = flow["bwd_pkt_lengths"] or [0]
    all_lengths = fwd_lengths + bwd_lengths
    flow_iats = flow["flow_iat"] or [0]
    bwd_iats = flow["bwd_iat"] or [0]

    duration = (flow["last_time"] - flow["start_time"]) if flow["start_time"] and flow["last_time"] else 0
    duration_us = max(duration * 1e6, 1)

    total_fwd = flow["fwd_packets"]
    total_bwd = flow["bwd_packets"]
    total_packets = total_fwd + total_bwd
    total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]

    features = np.zeros(n_features, dtype=np.float64)

    # Map features based on the CICDDoS2019 column order after preprocessing
    features[0] = flow["protocol"]                              # Protocol
    features[1] = duration_us                                   # Flow Duration
    features[2] = total_fwd                                     # Total Fwd Packets
    features[3] = total_bwd                                     # Total Backward Packets
    features[4] = flow["fwd_bytes"]                             # Fwd Packets Length Total
    features[5] = max(fwd_lengths)                              # Fwd Packet Length Max
    features[6] = min(fwd_lengths)                              # Fwd Packet Length Min
    features[7] = np.std(fwd_lengths) if len(fwd_lengths) > 1 else 0  # Fwd Packet Length Std
    features[8] = max(bwd_lengths)                              # Bwd Packet Length Max
    features[9] = min(bwd_lengths)                              # Bwd Packet Length Min
    features[10] = np.std(bwd_lengths) if len(bwd_lengths) > 1 else 0  # Bwd Packet Length Std
    features[11] = total_bytes / duration_us * 1e6 if duration_us > 0 else 0  # Flow Bytes/s
    features[12] = total_packets / duration_us * 1e6 if duration_us > 0 else 0  # Flow Packets/s
    features[13] = np.mean(flow_iats) * 1e6                     # Flow IAT Mean
    features[14] = min(flow_iats) * 1e6                         # Flow IAT Min
    features[15] = np.mean(bwd_iats) * 1e6 if bwd_iats != [0] else 0  # Bwd IAT Total
    features[16] = np.mean(bwd_iats) * 1e6 if bwd_iats != [0] else 0  # Bwd IAT Mean
    features[17] = min(bwd_iats) * 1e6 if bwd_iats != [0] else 0  # Bwd IAT Min
    features[18] = 1 if flow["flags"].get("PSH", 0) > 0 else 0  # Fwd PSH Flags
    features[19] = flow["flags"].get("SYN", 0)                  # SYN Flag Count
    features[20] = flow["flags"].get("ACK", 0)                  # ACK Flag Count
    features[21] = flow["flags"].get("URG", 0)                  # URG Flag Count
    features[22] = flow["flags"].get("CWE", 0)                  # CWE Flag Count
    features[23] = total_bwd / total_fwd if total_fwd > 0 else 0  # Down/Up Ratio
    features[24] = flow["fwd_bytes"] / max(total_fwd, 1)        # Fwd Header Length
    features[25] = flow["bwd_bytes"] / max(total_bwd, 1) if total_bwd > 0 else 0  # Bwd Header Length
    features[26] = total_fwd / duration_us * 1e6 if duration_us > 0 else 0  # Bwd Packets/s
    features[27] = np.mean(all_lengths)                         # Packet Length Mean (mapped)
    features[28] = np.mean(all_lengths)                         # Avg Packet Size (mapped)
    features[29] = flow["fwd_bytes"]                            # Init Fwd Win Bytes
    features[30] = flow["bwd_bytes"]                            # Init Bwd Win Bytes
    features[31] = np.mean([0]) if not flow.get("active_mean") else 0  # Active/Idle features

    return features


def flow_to_feature_frame(flow: dict) -> pd.DataFrame:
    """Convert a live flow into a CICDDoS-like raw feature DataFrame."""
    fwd_lengths = flow["fwd_pkt_lengths"] or [0]
    bwd_lengths = flow["bwd_pkt_lengths"] or [0]
    all_lengths = fwd_lengths + bwd_lengths
    flow_iats = flow["flow_iat"] or [0]
    fwd_iats = flow["fwd_iat"] or [0]
    bwd_iats = flow["bwd_iat"] or [0]

    duration = (flow["last_time"] - flow["start_time"]) if flow["start_time"] and flow["last_time"] else 0
    duration_us = max(duration * 1e6, 1)

    total_fwd = flow["fwd_packets"]
    total_bwd = flow["bwd_packets"]
    total_packets = total_fwd + total_bwd
    total_bytes = flow["fwd_bytes"] + flow["bwd_bytes"]
    active_times = [duration_us]
    idle_times = [iat * 1e6 for iat in flow_iats if iat >= 1.0] or [0.0]
    fwd_seg_sizes = flow.get("fwd_seg_sizes") or [0]
    init_fwd_win = flow.get("init_fwd_win_bytes", -1)
    init_bwd_win = flow.get("init_bwd_win_bytes", -1)

    row = {column: 0.0 for column in CICDDOS_FEATURE_COLUMNS}
    row.update({
        "Protocol": flow["protocol"],
        "Flow Duration": duration_us,
        "Total Fwd Packets": total_fwd,
        "Total Backward Packets": total_bwd,
        "Fwd Packets Length Total": flow["fwd_bytes"],
        "Bwd Packets Length Total": flow["bwd_bytes"],
        "Fwd Packet Length Max": max(fwd_lengths),
        "Fwd Packet Length Min": min(fwd_lengths),
        "Fwd Packet Length Mean": float(np.mean(fwd_lengths)),
        "Fwd Packet Length Std": float(np.std(fwd_lengths)) if len(fwd_lengths) > 1 else 0.0,
        "Bwd Packet Length Max": max(bwd_lengths),
        "Bwd Packet Length Min": min(bwd_lengths),
        "Bwd Packet Length Mean": float(np.mean(bwd_lengths)),
        "Bwd Packet Length Std": float(np.std(bwd_lengths)) if len(bwd_lengths) > 1 else 0.0,
        "Flow Bytes/s": total_bytes / duration_us * 1e6,
        "Flow Packets/s": total_packets / duration_us * 1e6,
        "Flow IAT Mean": float(np.mean(flow_iats)) * 1e6,
        "Flow IAT Std": float(np.std(flow_iats)) * 1e6 if len(flow_iats) > 1 else 0.0,
        "Flow IAT Max": max(flow_iats) * 1e6,
        "Flow IAT Min": min(flow_iats) * 1e6,
        "Fwd IAT Total": sum(fwd_iats) * 1e6,
        "Fwd IAT Mean": float(np.mean(fwd_iats)) * 1e6,
        "Fwd IAT Std": float(np.std(fwd_iats)) * 1e6 if len(fwd_iats) > 1 else 0.0,
        "Fwd IAT Max": max(fwd_iats) * 1e6,
        "Fwd IAT Min": min(fwd_iats) * 1e6,
        "Bwd IAT Total": sum(bwd_iats) * 1e6,
        "Bwd IAT Mean": float(np.mean(bwd_iats)) * 1e6,
        "Bwd IAT Std": float(np.std(bwd_iats)) * 1e6 if len(bwd_iats) > 1 else 0.0,
        "Bwd IAT Max": max(bwd_iats) * 1e6,
        "Bwd IAT Min": min(bwd_iats) * 1e6,
        "Fwd PSH Flags": 1 if flow["flags"].get("PSH", 0) > 0 else 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 1 if flow["flags"].get("URG", 0) > 0 else 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": flow.get("fwd_header_bytes", 0),
        "Bwd Header Length": flow.get("bwd_header_bytes", 0),
        "Fwd Packets/s": total_fwd / duration_us * 1e6,
        "Bwd Packets/s": total_bwd / duration_us * 1e6,
        "Packet Length Min": min(all_lengths),
        "Packet Length Max": max(all_lengths),
        "Packet Length Mean": float(np.mean(all_lengths)),
        "Packet Length Std": float(np.std(all_lengths)) if len(all_lengths) > 1 else 0.0,
        "Packet Length Variance": float(np.var(all_lengths)) if len(all_lengths) > 1 else 0.0,
        "FIN Flag Count": flow["flags"].get("FIN", 0),
        "SYN Flag Count": flow["flags"].get("SYN", 0),
        "RST Flag Count": flow["flags"].get("RST", 0),
        "PSH Flag Count": flow["flags"].get("PSH", 0),
        "ACK Flag Count": flow["flags"].get("ACK", 0),
        "URG Flag Count": flow["flags"].get("URG", 0),
        "CWE Flag Count": flow["flags"].get("CWE", 0),
        "ECE Flag Count": flow["flags"].get("ECE", 0),
        "Down/Up Ratio": total_bwd / total_fwd if total_fwd > 0 else 0,
        "Avg Packet Size": total_bytes / max(total_packets, 1),
        "Avg Fwd Segment Size": float(np.mean(fwd_lengths)),
        "Avg Bwd Segment Size": float(np.mean(bwd_lengths)),
        "Subflow Fwd Packets": total_fwd,
        "Subflow Fwd Bytes": flow["fwd_bytes"],
        "Subflow Bwd Packets": total_bwd,
        "Subflow Bwd Bytes": flow["bwd_bytes"],
        "Init Fwd Win Bytes": init_fwd_win if init_fwd_win >= 0 else 0,
        "Init Bwd Win Bytes": init_bwd_win if init_bwd_win >= 0 else 0,
        "Fwd Act Data Packets": flow.get("fwd_data_packets", 0),
        "Fwd Seg Size Min": min(fwd_seg_sizes),
        "Active Mean": float(np.mean(active_times)),
        "Active Std": float(np.std(active_times)) if len(active_times) > 1 else 0.0,
        "Active Max": max(active_times),
        "Active Min": min(active_times),
        "Idle Mean": float(np.mean(idle_times)),
        "Idle Std": float(np.std(idle_times)) if len(idle_times) > 1 else 0.0,
        "Idle Max": max(idle_times),
        "Idle Min": min(idle_times),
    })
    return pd.DataFrame([row], columns=CICDDOS_FEATURE_COLUMNS)


def predict_flow(model, flow: dict, label_map: dict):
    """Predict a flow with either the new Pipeline model or legacy 32-feature model."""
    feature_frame = flow_to_feature_frame(flow)
    try:
        proba = model.predict_proba(feature_frame)[0]
    except Exception:
        legacy = flow_to_features(flow).reshape(1, -1)
        proba = model.predict_proba(legacy)[0]

    pred_class = int(np.argmax(proba))
    pred_prob = float(proba[pred_class])
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = getattr(model.named_steps["model"], "classes_", None)
    if classes is not None and pred_class < len(classes):
        pred_label = classes[pred_class]
        pred_name = label_map.get(int(pred_label), str(pred_label)) if isinstance(pred_label, (int, np.integer)) else str(pred_label)
    else:
        pred_name = label_map.get(pred_class, f"Class {pred_class}")
    return pred_name, pred_prob, pred_class


def _first_value(row: pd.Series, candidates: list[str], default: Any = "") -> Any:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def _load_model_by_name(model_name: str):
    if model_name != DEFAULT_MODEL:
        raise ValueError(
            f"Only the CICDDoS2019 production model is supported in live demo mode: {DEFAULT_MODEL}.pkl"
        )
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _class_name_from_model(model, class_index: int, label_map: dict[int, str]) -> str:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = getattr(model.named_steps["model"], "classes_", None)
    if classes is not None and class_index < len(classes):
        pred_label = classes[class_index]
        if isinstance(pred_label, (int, np.integer)):
            return label_map.get(int(pred_label), str(pred_label))
        return str(pred_label)
    return label_map.get(class_index, f"Class {class_index}")


def _run_cicflowmeter_python_api(pcap_path: str, csv_path: str) -> str:
    """Run the Python cicflowmeter package directly.

    cicflowmeter 0.5.0 has a CLI argument-order bug for offline PCAP mode.
    Calling create_sniffer with keyword arguments avoids that bug.
    """
    sniffer_module = importlib.import_module("cicflowmeter.sniffer")
    create_sniffer = getattr(sniffer_module, "create_sniffer")

    sniffer, session = create_sniffer(
        input_file=pcap_path,
        input_interface=None,
        output_mode="csv",
        output=csv_path,
        input_directory=None,
        fields=None,
        verbose=False,
    )
    sniffer.start()
    try:
        sniffer.join()
    finally:
        if hasattr(session, "_gc_stop"):
            session._gc_stop.set()
            session._gc_thread.join(timeout=2.0)
        sniffer.join()
        session.flush_flows()

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise RuntimeError("CICFlowMeter Python API did not produce a non-empty CSV.")
    return csv_path


def _run_cicflowmeter_command(command_template: str, pcap_path: str, csv_path: str, pcap_dir: str, csv_dir: str) -> str:
    def resolve_cli(command: str) -> str:
        parts = shlex.split(command)
        if not parts or parts[0] != "cicflowmeter":
            return command

        candidates = [
            shutil.which("cicflowmeter"),
            os.path.join(sys.prefix, "bin", "cicflowmeter"),
            os.path.join(os.getcwd(), "venv", "bin", "cicflowmeter"),
            os.path.join(os.getcwd(), "venv", "Scripts", "cicflowmeter.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                parts[0] = candidate
                return " ".join(shlex.quote(part) for part in parts)
        return command

    command = command_template.format(
        pcap=shlex.quote(pcap_path),
        csv=shlex.quote(csv_path),
        pcap_dir=shlex.quote(pcap_dir),
        csv_dir=shlex.quote(csv_dir),
    )
    raw_parts = shlex.split(command)
    if raw_parts and os.path.basename(raw_parts[0]) == "cicflowmeter":
        logger.info("Running CICFlowMeter through Python API: %s -> %s", pcap_path, csv_path)
        return _run_cicflowmeter_python_api(pcap_path, csv_path)

    command = resolve_cli(command)
    logger.info("Running CICFlowMeter: %s", command)
    result = subprocess.run(
        command,
        shell=True,
        cwd=pcap_dir,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "CICFlowMeter failed")

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return csv_path

    csv_files = [
        os.path.join(csv_dir, name)
        for name in os.listdir(csv_dir)
        if name.lower().endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"CICFlowMeter did not produce a CSV in {csv_dir}")
    return max(csv_files, key=os.path.getmtime)


def _normalize_cicflowmeter_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize Python CICFlowMeter output to the CICDDoS2019 feature schema."""
    working = df.copy()
    if "Label" in working.columns:
        working = working.drop(columns=["Label"])
    working.columns = [str(col).strip() for col in working.columns]

    display_frame = working.copy()
    model_frame = working.rename(columns=CICFLOWMETER_TO_CICDDOS_COLUMNS)
    for column in CICDDOS_FEATURE_COLUMNS:
        if column not in model_frame.columns:
            model_frame[column] = 0.0
    model_frame = model_frame[CICDDOS_FEATURE_COLUMNS]
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan)
    model_frame = model_frame.apply(pd.to_numeric, errors="coerce")
    for column in CICFLOW_TIME_COLUMNS_US:
        if column in model_frame.columns:
            model_frame[column] = model_frame[column] * 1_000_000.0

    return model_frame, display_frame


def _predict_cicflowmeter_rows(model, df: pd.DataFrame, label_map: dict[int, str], threshold: float) -> list[dict]:
    """Predict CICFlowMeter CSV rows with the trained CICDDoS2019 pipeline."""
    if df.empty:
        return []

    model_frame, display_frame = _normalize_cicflowmeter_features(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_frame.to_csv(os.path.join(RESULTS_DIR, "last_cicflowmeter_features.csv"), index=False)
    display_frame.to_csv(os.path.join(RESULTS_DIR, "last_cicflowmeter_raw.csv"), index=False)

    syn_total = float(model_frame["SYN Flag Count"].sum()) if "SYN Flag Count" in model_frame else 0.0
    duration_mean = float(model_frame["Flow Duration"].mean()) if "Flow Duration" in model_frame else 0.0
    logger.info(
        "CICFlowMeter rows=%s, normalized_features=%s, syn_flags_total=%.0f, flow_duration_mean_us=%.0f",
        len(model_frame),
        len(model_frame.columns),
        syn_total,
        duration_mean,
    )

    proba = model.predict_proba(model_frame)

    events = []
    for row_index, row in display_frame.reset_index(drop=True).iterrows():
        class_index = int(np.argmax(proba[row_index]))
        pred_prob = float(proba[row_index][class_index])
        pred_name = _class_name_from_model(model, class_index, label_map)
        ranked = sorted(
            [
                (_class_name_from_model(model, idx, label_map), float(prob))
                for idx, prob in enumerate(proba[row_index])
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        logger.info(
            "ML probabilities top3: %s",
            ", ".join(f"{name}={prob:.2%}" for name, prob in ranked),
        )
        is_attack = pred_name != "Benign" and pred_prob >= threshold
        display_prediction = pred_name if is_attack or pred_name == "Benign" else "Benign"
        src_ip = str(_first_value(row, ["Src IP", "Source IP", "src_ip", "SourceIP"], ""))
        dst_ip = str(_first_value(row, ["Dst IP", "Destination IP", "dst_ip", "DestinationIP"], ""))
        protocol = int(float(_first_value(row, ["Protocol", "protocol"], 0) or 0))
        src_port = int(float(_first_value(row, ["Src Port", "Source Port", "src_port"], 0) or 0))
        dst_port = int(float(_first_value(row, ["Dst Port", "Destination Port", "dst_port"], 0) or 0))
        fwd_packets = int(float(_first_value(row, ["Total Fwd Packets", "Tot Fwd Pkts", "tot_fwd_pkts"], 0) or 0))
        bwd_packets = int(float(_first_value(row, ["Total Backward Packets", "Tot Bwd Pkts", "tot_bwd_pkts"], 0) or 0))
        events.append({
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": protocol,
            "src_port": src_port,
            "dst_port": dst_port,
            "dst_ports": f"{dst_port}:{fwd_packets}" if dst_port else "",
            "fwd_packets": fwd_packets,
            "bwd_packets": bwd_packets,
            "prediction": display_prediction,
            "confidence": pred_prob,
            "is_attack": is_attack,
        })
    return events


def run_cicflowmeter_ips(
    model_name: str,
    interface: str | None,
    victim_ip: str | None,
    threshold: float,
    simulation: bool,
    mitigation_backend: str,
    events_csv: str,
    cicflowmeter_cmd: str,
    window_seconds: float,
    windows: int,
) -> None:
    """Capture real packets, extract CICFlowMeter CSV features, and classify with selected_model.pkl."""
    if not cicflowmeter_cmd:
        raise ValueError("Missing --cicflowmeter-cmd or CICFLOWMETER_CMD environment variable.")

    label_map = {
        0: "Benign", 1: "LDAP Flood", 2: "MSSQL Flood",
        3: "NetBIOS Flood", 4: "TCP SYN Flood",
        5: "UDP Flood", 6: "UDP-Lag Flood",
    }
    model = _load_model_by_name(model_name)
    event_writer = LiveIPS(
        model_name=model_name,
        interface=interface,
        simulation=simulation,
        threshold=threshold,
        mitigation_backend=mitigation_backend,
        events_csv=events_csv,
    )
    event_writer.model = model

    logger.info(
        "CICFlowMeter mode enabled. Model=%s, interface=%s, victim_ip=%s",
        model_name,
        interface or "auto",
        victim_ip or "not-filtered",
    )
    logger.info("Capture window: %.1fs", window_seconds)
    if window_seconds > 5:
        logger.info("Tip: use --fast-log or --cic-window 2 for faster dashboard updates during demo.")
    completed_windows = 0

    with tempfile.TemporaryDirectory(prefix="mlddos_cicflow_") as temp_dir:
        pcap_dir = os.path.join(temp_dir, "pcap")
        csv_dir = os.path.join(temp_dir, "csv")
        os.makedirs(pcap_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        while windows <= 0 or completed_windows < windows:
            completed_windows += 1
            capture_filter = f"ip and dst host {victim_ip}" if victim_ip else "ip"
            logger.info("Capturing window %s with filter: %s", completed_windows, capture_filter)
            packets = sniff(
                iface=interface,
                store=True,
                timeout=window_seconds,
                filter=capture_filter,
            )
            if not packets:
                logger.info("No packets captured in this window.")
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pcap_path = os.path.join(pcap_dir, f"window_{stamp}.pcap")
            csv_path = os.path.join(csv_dir, f"window_{stamp}.csv")
            wrpcap(pcap_path, packets)

            try:
                produced_csv = _run_cicflowmeter_command(
                    cicflowmeter_cmd,
                    pcap_path,
                    csv_path,
                    pcap_dir,
                    csv_dir,
                )
                flow_df = pd.read_csv(produced_csv)
                events = _predict_cicflowmeter_rows(model, flow_df, label_map, threshold)
                if victim_ip:
                    events = [event for event in events if str(event.get("dst_ip", "")) == victim_ip]
            except Exception as exc:
                logger.error("CICFlowMeter prediction window failed: %s", exc)
                continue

            for event in events:
                event.update({
                    "timestamp": datetime.now().isoformat(),
                    "blocked": False,
                    "simulation": simulation,
                    "model": model_name,
                    "interface": interface or "auto",
                    "victim_os": event_writer.mitigation_status.get("os", ""),
                    "mitigation_backend": event_writer.mitigation_status.get("backend", ""),
                    "mitigation_elevated": event_writer.mitigation_status.get("elevated", False),
                    "block_command": "",
                    "block_message": "",
                    "source": "live_ips",
                    "decision_source": "cicflowmeter_ml_model",
                })

                if event["is_attack"]:
                    logger.warning(
                        "[ALERT] %s -> %s | %s (%.2f%%)",
                        event["src_ip"],
                        event["dst_ip"],
                        event["prediction"],
                        event["confidence"] * 100,
                    )
                    if event["confidence"] >= threshold and event["src_ip"] not in event_writer.blocked_ips:
                        result = block_ip(
                            event["src_ip"],
                            reason=f"{event['prediction']} (confidence: {event['confidence']:.2%})",
                            simulation=simulation,
                            backend=mitigation_backend,
                        )
                        event["block_command"] = result.get("command", "")
                        event["block_message"] = result.get("message", "")
                        event["blocked"] = bool(result.get("success", False))
                        if event["blocked"]:
                            event_writer.blocked_ips.add(event["src_ip"])
                            show_ddos_popup(event)
                else:
                    logger.info(
                        "[OK] %s -> %s | Benign (%.2f%%)",
                        event["src_ip"],
                        event["dst_ip"],
                        event["confidence"] * 100,
                    )
                event_writer._write_event(event)
            logger.info("Wrote %s event(s) to %s", len(events), events_csv)


def collect_live_lab_dataset(
    label: str,
    interface: str | None,
    victim_ip: str | None,
    cicflowmeter_cmd: str,
    window_seconds: float,
    windows: int,
    dataset_csv: str,
) -> None:
    """Capture live lab traffic and append normalized CICFlowMeter rows with a label."""
    if not cicflowmeter_cmd:
        raise ValueError("Missing --cicflowmeter-cmd or CICFLOWMETER_CMD environment variable.")
    if windows <= 0:
        raise ValueError("--cic-windows must be greater than 0 when collecting labeled live data.")

    os.makedirs(os.path.dirname(dataset_csv) or ".", exist_ok=True)
    logger.info("Collecting live-lab label=%s into %s", label, dataset_csv)

    with tempfile.TemporaryDirectory(prefix="mlddos_live_lab_") as temp_dir:
        pcap_dir = os.path.join(temp_dir, "pcap")
        csv_dir = os.path.join(temp_dir, "csv")
        os.makedirs(pcap_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        total_rows = 0
        for window_index in range(1, windows + 1):
            capture_filter = f"ip and dst host {victim_ip}" if victim_ip else "ip"
            logger.info("Collecting window %s/%s with filter: %s", window_index, windows, capture_filter)
            packets = sniff(
                iface=interface,
                store=True,
                timeout=window_seconds,
                filter=capture_filter,
            )
            if not packets:
                logger.info("No packets captured in this window.")
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pcap_path = os.path.join(pcap_dir, f"collect_{stamp}.pcap")
            csv_path = os.path.join(csv_dir, f"collect_{stamp}.csv")
            wrpcap(pcap_path, packets)

            try:
                produced_csv = _run_cicflowmeter_command(
                    cicflowmeter_cmd,
                    pcap_path,
                    csv_path,
                    pcap_dir,
                    csv_dir,
                )
                flow_df = pd.read_csv(produced_csv)
                model_frame, display_frame = _normalize_cicflowmeter_features(flow_df)
                if victim_ip:
                    dst_values = display_frame.apply(
                        lambda row: str(_first_value(row, ["Dst IP", "Destination IP", "dst_ip", "DestinationIP"], "")),
                        axis=1,
                    )
                    model_frame = model_frame.loc[dst_values == victim_ip].copy()
                if model_frame.empty:
                    logger.info("No normalized rows kept for this window.")
                    continue
                model_frame["Label"] = label
                model_frame.to_csv(
                    dataset_csv,
                    mode="a",
                    index=False,
                    header=not os.path.exists(dataset_csv),
                )
                total_rows += len(model_frame)
                logger.info("Appended %s rows for label=%s. Total this run=%s", len(model_frame), label, total_rows)
            except Exception as exc:
                logger.error("Live-lab collection window failed: %s", exc)

    logger.info("Finished live-lab collection. Rows appended in this run: %s", total_rows)


def _load_cicddos_balanced_sample(max_per_class: int) -> pd.DataFrame:
    """Load a balanced CICDDoS2019 sample using the same deployment feature schema."""
    from .preprocessor import DDOS_CATEGORY_MAP

    parquet_paths = [
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.endswith(".parquet") and "-training" in name
    ]
    if not parquet_paths:
        raise FileNotFoundError(f"No CICDDoS2019 training parquet files found in {DATA_DIR}")

    frames = []
    for path in sorted(parquet_paths):
        df = pd.read_parquet(path)
        if "Label" not in df.columns:
            continue
        missing = [column for column in CICDDOS_FEATURE_COLUMNS if column not in df.columns]
        if missing:
            logger.warning("Skipping %s because required feature columns are missing: %s", path, missing[:5])
            continue
        df = df[CICDDOS_FEATURE_COLUMNS + ["Label"]].copy()
        df["Label"] = df["Label"].map(lambda value: DDOS_CATEGORY_MAP.get(value, value))
        df = df.dropna(subset=["Label"])
        df = df[df["Label"].isin(DEPLOYMENT_LABELS)]
        frames.append(df)

    if not frames:
        raise ValueError("No usable CICDDoS2019 parquet rows found.")

    combined = pd.concat(frames, ignore_index=True)
    samples = []
    for label, group in combined.groupby("Label"):
        n = min(len(group), max_per_class)
        samples.append(group.sample(n=n, random_state=42))
    sampled = pd.concat(samples, ignore_index=True)
    logger.info("CICDDoS2019 sampled distribution:\n%s", sampled["Label"].value_counts().to_string())
    return sampled


def train_live_lab_model(
    dataset_csv: str,
    output_model: str,
    test_size: float,
    mix_cicddos: bool = True,
    cic_samples_per_class: int = 3000,
    live_sample_weight: float = 10.0,
) -> None:
    """Train selected_model.pkl from CICDDoS2019 plus labeled live-lab features."""
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Live-lab dataset not found: {dataset_csv}")

    live_df = pd.read_csv(dataset_csv)
    if "Label" not in live_df.columns:
        raise ValueError("Live-lab dataset must contain a Label column.")
    missing = [column for column in CICDDOS_FEATURE_COLUMNS if column not in live_df.columns]
    if missing:
        raise ValueError(f"Live-lab dataset is missing feature columns: {missing[:10]}")

    live_df = live_df[CICDDOS_FEATURE_COLUMNS + ["Label"]].dropna(subset=["Label"]).copy()
    dropped_live = live_df[~live_df["Label"].isin(DEPLOYMENT_LABELS)]
    if not dropped_live.empty:
        logger.info(
            "Dropping unsupported live-lab labels from deployment training: %s",
            sorted(dropped_live["Label"].unique()),
        )
    live_df = live_df[live_df["Label"].isin(DEPLOYMENT_LABELS)].copy()
    if live_df.empty:
        raise ValueError("No supported deployment labels remain after filtering live-lab dataset.")
    live_df["_source"] = "live_lab"
    live_counts = live_df["Label"].value_counts()
    logger.info("Live-lab class distribution:\n%s", live_counts.to_string())

    frames = [live_df]
    if mix_cicddos:
        cic_df = _load_cicddos_balanced_sample(cic_samples_per_class)
        cic_df["_source"] = "cicddos2019"
        frames.insert(0, cic_df)

    df = pd.concat(frames, ignore_index=True)
    class_counts = df["Label"].value_counts()
    logger.info("Mixed training distribution:\n%s", class_counts.to_string())
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes to train a deployment model.")
    if live_counts.min() < 5:
        logger.warning("Some live-lab classes have fewer than 5 samples. Collect more windows for stable demo accuracy.")

    X = df[CICDDOS_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df["Label"].astype(str)
    sample_weights = np.where(df["_source"].eq("live_lab"), float(live_sample_weight), 1.0)

    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test, w_train, _w_test = train_test_split(
        X,
        y,
        sample_weights,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("model", ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train, model__sample_weight=w_train)

    y_pred = model.predict(X_test)
    logger.info("Live-lab holdout accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info(
        "Live-lab classification report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{output_model}.pkl")
    if output_model == DEFAULT_MODEL and os.path.exists(model_path):
        backup_path = os.path.join(
            MODELS_DIR,
            f"{DEFAULT_MODEL}_cicddos_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        )
        shutil.copy2(model_path, backup_path)
        logger.info("Backed up previous selected model: %s", backup_path)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    logger.info("Saved live-lab model: %s", model_path)


class LiveIPS:
    """Real-time Intrusion Prevention System engine."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 interface: str | None = None,
                 simulation: bool = True,
                 threshold: float = ATTACK_THRESHOLD,
                 mitigation_backend: str = "auto",
                 callback=None,
                 events_csv: str = DEFAULT_EVENTS_CSV):
        """
        Parameters
        ----------
        model_name : str
            Name of the saved model to load.
        interface : str
            Network interface to sniff on (None = default).
        simulation : bool
            If True, block commands are printed only.
        threshold : float
            Attack probability threshold for triggering mitigation.
        callback : callable or None
            Optional callback(event_dict) for each detection event.
        events_csv : str
            CSV path used by the dashboard to read real live IPS events.
        """
        self.model_name = model_name
        self.interface = interface
        self.simulation = simulation
        self.threshold = threshold
        self.mitigation_backend = mitigation_backend
        self.mitigation_status = get_mitigation_status(mitigation_backend)
        self.callback = callback
        self.events_csv = events_csv
        self.capture_label: str | None = None

        self.model = None
        self.scaler = None
        self.label_map = {}
        self.aggregator = FlowAggregator()
        self.running = False
        self.blocked_ips = set()
        self.event_log = []

        self._load_model()
        self._ensure_events_csv()

    def _ensure_events_csv(self):
        """Create the event CSV with a stable header for the dashboard."""
        if not self.events_csv:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.events_csv)), exist_ok=True)
        if os.path.exists(self.events_csv) and os.path.getsize(self.events_csv) > 0:
            try:
                with open(self.events_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, [])
                if existing_header == self._event_fields():
                    return
                logger.warning("Existing events CSV has an old schema. Recreating: %s", self.events_csv)
            except Exception:
                logger.warning("Could not read events CSV header. Recreating: %s", self.events_csv)

        with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._event_fields())
            writer.writeheader()

    @staticmethod
    def _event_fields():
        return [
            "timestamp", "src_ip", "dst_ip", "protocol",
            "src_port", "dst_port", "dst_ports",
            "fwd_packets", "bwd_packets", "prediction", "confidence",
            "decision_source",
            "is_attack", "blocked", "simulation", "model", "interface",
            "victim_os", "mitigation_backend", "mitigation_elevated",
            "block_command", "block_message", "source",
        ]

    def _write_event(self, event: dict):
        """Append one IPS event so Streamlit can display real live detections."""
        if not self.events_csv:
            return

        row = {field: event.get(field, "") for field in self._event_fields()}
        with open(self.events_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._event_fields())
            writer.writerow(row)

    def _load_model(self):
        """Load the trained model and scaler."""
        if self.model_name != DEFAULT_MODEL:
            logger.warning(
                "Only %s.pkl is supported for the live CICDDoS2019 demo. Ignoring requested model: %s",
                DEFAULT_MODEL,
                self.model_name,
            )
            self.model_name = DEFAULT_MODEL
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.info("Run 'python main.py' first to train models.")
            return

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model: {self.model_name}")

        # Load label map if available
        self.label_map = {
            0: "Benign", 1: "LDAP Flood", 2: "MSSQL Flood",
            3: "NetBIOS Flood", 4: "TCP SYN Flood",
            5: "UDP Flood", 6: "UDP-Lag Flood",
        }

    def _packet_callback(self, packet):
        """Callback for each sniffed packet."""
        self.aggregator.add_packet(packet)

    def _analyze_flows(self):
        """Periodically check completed flows and classify them."""
        while self.running:
            time.sleep(FLOW_CHECK_INTERVAL)
            completed = self.aggregator.get_completed_flows()

            for key, flow in completed.items():
                try:
                    pred_name, pred_prob, pred_class = predict_flow(
                        self.model,
                        flow,
                        self.label_map,
                    )
                    src_ip = flow["src_ip"]
                    dst_ip = flow["dst_ip"]
                    dst_ports = flow.get("dst_ports", {})
                    dst_ports_summary = ";".join(
                        f"{int(port)}:{int(count)}"
                        for port, count in sorted(
                            dst_ports.items(),
                            key=lambda item: int(item[1]),
                            reverse=True,
                        )[:5]
                    )

                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": flow["protocol"],
                        "src_port": flow.get("src_port", 0),
                        "dst_port": flow.get("dst_port", 0),
                        "dst_ports": dst_ports_summary,
                        "fwd_packets": flow["fwd_packets"],
                        "bwd_packets": flow["bwd_packets"],
                        "prediction": pred_name,
                        "confidence": float(pred_prob),
                        "decision_source": "ml_model",
                        "is_attack": pred_name != "Benign",
                        "blocked": False,
                        "simulation": self.simulation,
                        "model": self.model_name,
                        "interface": self.interface or "auto",
                        "victim_os": self.mitigation_status.get("os", ""),
                        "mitigation_backend": self.mitigation_status.get("backend", ""),
                        "mitigation_elevated": self.mitigation_status.get("elevated", False),
                        "block_command": "",
                        "block_message": "",
                        "source": "live_ips",
                    }

                    if pred_name == "Benign":
                        logger.info(
                            f"[OK] {src_ip} -> {dst_ip} | "
                            f"Benign ({pred_prob:.2%})"
                        )
                    elif self.capture_label:
                        logger.info(
                            f"[CAPTURE] {src_ip} -> {dst_ip} | "
                            f"saved as {self.capture_label}"
                        )
                    else:
                        logger.warning(
                            f"[ALERT] {src_ip} -> {dst_ip} | "
                            f"{pred_name} ({pred_prob:.2%})"
                        )

                        if pred_prob >= self.threshold and src_ip not in self.blocked_ips:
                            logger.critical(
                                f"[IPS] Blocking {src_ip} — {pred_name} "
                                f"(confidence: {pred_prob:.2%})"
                            )
                            
                                    # Step 5.1: show popup immediately after DDoS detection
                            show_ddos_popup(event)

                            # Step 5.2: block attacker IP
                            logger.critical(
                                f"[IPS] Blocking attacker IP: {src_ip}"
                          )
                            result = block_ip(
                                src_ip,
                                reason=f"{pred_name} (confidence: {pred_prob:.2%})",
                                simulation=self.simulation,
                                backend=self.mitigation_backend,
                            )
                            event["block_command"] = result.get("command", "")
                            event["block_message"] = result.get("message", "")
                            event["mitigation_backend"] = result.get(
                                "backend",
                                event["mitigation_backend"],
                            )
                            event["mitigation_elevated"] = result.get(
                                "elevated",
                                event["mitigation_elevated"],
                            )
                            if result["success"]:
                                self.blocked_ips.add(src_ip)
                                event["blocked"] = True

                    self.event_log.append(event)
                    self._write_event(event)
                    if self.callback:
                        self.callback(event)

                except Exception as e:
                    logger.error(f"Error analyzing flow {key}: {e}")

    def start(self, count: int = 0):
        """Start the IPS engine.

        Parameters
        ----------
        count : int
            Number of packets to capture (0 = unlimited).
        """
        if self.model is None:
            logger.error("No model loaded. Cannot start IPS.")
            return

        self.running = True
        mode = "SIMULATION" if self.simulation else "LIVE"
        logger.info(f"Starting IPS [{mode} MODE] on interface: {self.interface or 'default'}")
        logger.info(f"Attack threshold: {self.threshold:.0%}")
        logger.info(
            "Mitigation backend: %s on %s (elevated=%s)",
            self.mitigation_status.get("backend"),
            self.mitigation_status.get("os"),
            self.mitigation_status.get("elevated"),
        )

        # Start flow analysis thread
        analysis_thread = threading.Thread(target=self._analyze_flows, daemon=True)
        analysis_thread.start()

        try:
            sniff(
                iface=self.interface,
                prn=self._packet_callback,
                store=False,
                count=count,
                filter="ip",
            )
        except PermissionError:
            logger.error("Permission denied. Run with sudo/root privileges.")
        except KeyboardInterrupt:
            logger.info("IPS stopped by user.")
        finally:
            self.running = False

    def stop(self):
        """Stop the IPS engine."""
        self.running = False
        logger.info("IPS engine stopped.")


def main():
    parser = argparse.ArgumentParser(description="Live DDoS IPS Engine")
    parser.add_argument("--list-interfaces", action="store_true",
                        help="List available Scapy capture interfaces and exit")
    parser.add_argument("--interface", "-i", type=str, default=None,
                        help="Network interface to sniff (default: auto)")
    parser.add_argument("--victim-ip", type=str, default=None,
                        help="Optional Victim IP filter. In CICFlowMeter mode, only flows whose destination matches this IP are evaluated.")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", "-t", type=float, default=ATTACK_THRESHOLD,
                        help="Attack probability threshold (default: 0.95)")
    parser.add_argument("--live", action="store_true",
                        help="DANGEROUS: Disable simulation mode (actually block IPs)")
    parser.add_argument("--mitigation-backend", default="auto",
                        choices=["auto", "iptables", "nftables", "windows_firewall", "manual"],
                        help="Firewall backend for blocking attacker IPs (default: auto)")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="Number of packets to capture (0=unlimited)")
    parser.add_argument("--events-csv", type=str, default=DEFAULT_EVENTS_CSV,
                        help=f"Path to write live IPS events (default: {DEFAULT_EVENTS_CSV})")
    parser.add_argument("--clear-events", action="store_true",
                        help="Clear the live events CSV before starting. Useful before recording a demo.")
    parser.add_argument("--cicflowmeter", action="store_true",
                        help="Use CICFlowMeter extraction from real captured PCAP windows before ML prediction.")
    parser.add_argument("--cicflowmeter-cmd", default=os.getenv("CICFLOWMETER_CMD", ""),
                        help=(
                            "Command template used to run CICFlowMeter. Supported placeholders: "
                            "{pcap}, {csv}, {pcap_dir}, {csv_dir}. Can also be set via CICFLOWMETER_CMD."
                        ))
    parser.add_argument("--cic-window", type=float, default=10.0,
                        help="Seconds per CICFlowMeter capture window (default: 10)")
    parser.add_argument("--fast-log", action="store_true",
                        help="Shortcut for demo: use 2-second CICFlowMeter windows for faster dashboard/log updates.")
    parser.add_argument("--cic-windows", type=int, default=0,
                        help="Number of CICFlowMeter windows to process (0=unlimited)")
    parser.add_argument("--live-dataset", type=str,
                        default=os.path.join(RESULTS_DIR, "live_lab_dataset.csv"),
                        help="CSV path used by live-lab collection/training")
    parser.add_argument("--collect-live-label", type=str, default=None,
                        help="Collect labeled live-lab CICFlowMeter rows for this class label, then exit")
    parser.add_argument("--train-live-lab-model", action="store_true",
                        help="Train a deployment model from --live-dataset, then exit")
    parser.add_argument("--live-model-output", type=str, default=DEFAULT_MODEL,
                        help=f"Output model name for live-lab training (default: {DEFAULT_MODEL})")
    parser.add_argument("--live-test-size", type=float, default=0.25,
                        help="Holdout ratio for live-lab training evaluation (default: 0.25)")
    parser.add_argument("--no-mix-cicddos", action="store_true",
                        help="Train only from live-lab dataset. Not recommended unless enough live data is available.")
    parser.add_argument("--cic-samples-per-class", type=int, default=3000,
                        help="Maximum CICDDoS2019 rows per class to mix into live-lab training (default: 3000)")
    parser.add_argument("--live-sample-weight", type=float, default=10.0,
                        help="Training weight for live-lab rows when mixed with CICDDoS2019 (default: 10)")
    args = parser.parse_args()

    if args.list_interfaces:
        print("Available capture interfaces:")
        for iface in get_if_list():
            print(f"  - {iface}")
        return

    simulation = not args.live

    if args.fast_log:
        args.cic_window = min(float(args.cic_window), 2.0)

    if args.clear_events and args.events_csv:
        try:
            if os.path.exists(args.events_csv):
                os.remove(args.events_csv)
                logger.info("Cleared events CSV: %s", args.events_csv)
        except OSError as exc:
            logger.warning("Could not clear events CSV %s: %s", args.events_csv, exc)

    if args.collect_live_label:
        collect_live_lab_dataset(
            label=args.collect_live_label,
            interface=args.interface,
            victim_ip=args.victim_ip,
            cicflowmeter_cmd=args.cicflowmeter_cmd,
            window_seconds=args.cic_window,
            windows=args.cic_windows,
            dataset_csv=args.live_dataset,
        )
        return

    if args.train_live_lab_model:
        train_live_lab_model(
            dataset_csv=args.live_dataset,
            output_model=args.live_model_output,
            test_size=args.live_test_size,
            mix_cicddos=not args.no_mix_cicddos,
            cic_samples_per_class=args.cic_samples_per_class,
            live_sample_weight=args.live_sample_weight,
        )
        return

    if args.cicflowmeter:
        run_cicflowmeter_ips(
            model_name=args.model,
            interface=args.interface,
            victim_ip=args.victim_ip,
            threshold=args.threshold,
            simulation=simulation,
            mitigation_backend=args.mitigation_backend,
            events_csv=args.events_csv,
            cicflowmeter_cmd=args.cicflowmeter_cmd,
            window_seconds=args.cic_window,
            windows=args.cic_windows,
        )
        return

    ips = LiveIPS(
        model_name=args.model,
        interface=args.interface,
        simulation=simulation,
        threshold=args.threshold,
        mitigation_backend=args.mitigation_backend,
        events_csv=args.events_csv,
    )

    mitigation_status = get_mitigation_status(args.mitigation_backend)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           DDoS IPS ENGINE — {'SIMULATION' if simulation else '  LIVE   '} MODE              ║
║                                                          ║
║  Model:     {args.model:<44s}║
║  Threshold: {args.threshold:<44.0%}║
║  Interface: {str(args.interface or 'auto'):<44s}║
║                                                          ║
║  {'[SAFE] Commands will be PRINTED only' if simulation else '[DANGER] Commands will be EXECUTED!':<56s}║
╚══════════════════════════════════════════════════════════╝
""")

    ips.start(count=args.count)


if __name__ == "__main__":
    main()
