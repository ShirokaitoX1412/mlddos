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
import logging
import threading
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scapy.all import sniff, IP, TCP, UDP, get_if_list
except ImportError:
    print("[live_ips] Scapy not installed. Install: pip install scapy")
    sys.exit(1)

from .mitigation import block_ip
from .alert_notifier import show_ddos_popup
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
            "src_ip": "",
            "dst_ip": "",
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
        pkt_len = len(packet)

        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

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
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += pkt_len
                flow["bwd_pkt_lengths"].append(pkt_len)

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

    def get_completed_flows(self, timeout: float = FLOW_TIMEOUT):
        """Return flows that have been idle for longer than timeout."""
        now = time.time()
        completed = {}

        with self.lock:
            expired_keys = []
            for key, flow in self.flows.items():
                if flow["last_time"] and (now - flow["last_time"]) > timeout:
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
        "Fwd URG Flags": 1 if flow["flags"].get("URG", 0) > 0 else 0,
        "Fwd Header Length": flow["fwd_bytes"] / max(total_fwd, 1),
        "Bwd Header Length": flow["bwd_bytes"] / max(total_bwd, 1) if total_bwd > 0 else 0,
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
        "Down/Up Ratio": total_bwd / total_fwd if total_fwd > 0 else 0,
        "Avg Packet Size": total_bytes / max(total_packets, 1),
        "Avg Fwd Segment Size": float(np.mean(fwd_lengths)),
        "Avg Bwd Segment Size": float(np.mean(bwd_lengths)),
        "Subflow Fwd Packets": total_fwd,
        "Subflow Fwd Bytes": flow["fwd_bytes"],
        "Subflow Bwd Packets": total_bwd,
        "Subflow Bwd Bytes": flow["bwd_bytes"],
        "Init Fwd Win Bytes": flow["fwd_bytes"],
        "Init Bwd Win Bytes": flow["bwd_bytes"],
        "Fwd Act Data Packets": total_fwd,
        "Fwd Seg Size Min": min(fwd_lengths),
        "Active Mean": duration_us,
        "Active Max": duration_us,
        "Active Min": duration_us,
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
    pred_name = label_map.get(pred_class, f"Class {pred_class}")
    return pred_name, pred_prob, pred_class


class LiveIPS:
    """Real-time Intrusion Prevention System engine."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 interface: str = None,
                 simulation: bool = True,
                 threshold: float = ATTACK_THRESHOLD,
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
        self.callback = callback
        self.events_csv = events_csv

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
            return

        with open(self.events_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._event_fields())
            writer.writeheader()

    @staticmethod
    def _event_fields():
        return [
            "timestamp", "src_ip", "dst_ip", "protocol",
            "fwd_packets", "bwd_packets", "prediction", "confidence",
            "is_attack", "blocked", "simulation", "model", "interface",
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
        model_path = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")
        if not os.path.exists(model_path) and self.model_name == "selected_model":
            fallback = os.path.join(MODELS_DIR, "random_forest.pkl")
            if os.path.exists(fallback):
                logger.warning("selected_model.pkl not found. Falling back to random_forest.pkl")
                model_path = fallback
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

                    event = {
                        "timestamp": datetime.now().isoformat(),
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": flow["protocol"],
                        "fwd_packets": flow["fwd_packets"],
                        "bwd_packets": flow["bwd_packets"],
                        "prediction": pred_name,
                        "confidence": float(pred_prob),
                        "is_attack": pred_name != "Benign",
                        "blocked": False,
                        "simulation": self.simulation,
                        "model": self.model_name,
                        "interface": self.interface or "auto",
                    }

                    if pred_name == "Benign":
                        logger.info(
                            f"[OK] {src_ip} -> {dst_ip} | "
                            f"Benign ({pred_prob:.2%})"
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
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", "-t", type=float, default=ATTACK_THRESHOLD,
                        help="Attack probability threshold (default: 0.95)")
    parser.add_argument("--live", action="store_true",
                        help="DANGEROUS: Disable simulation mode (actually block IPs)")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="Number of packets to capture (0=unlimited)")
    parser.add_argument("--events-csv", type=str, default=DEFAULT_EVENTS_CSV,
                        help=f"Path to write live IPS events (default: {DEFAULT_EVENTS_CSV})")
    args = parser.parse_args()

    if args.list_interfaces:
        print("Available capture interfaces:")
        for iface in get_if_list():
            print(f"  - {iface}")
        return

    simulation = not args.live

    ips = LiveIPS(
        model_name=args.model,
        interface=args.interface,
        simulation=simulation,
        threshold=args.threshold,
        events_csv=args.events_csv,
    )

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
