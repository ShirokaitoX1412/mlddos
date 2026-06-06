"""
Ryu SDN controller for ML-based DDoS IDS/IPS demonstration.

The controller works as an OpenFlow 1.3 learning switch and periodically
queries flow statistics from Open vSwitch. Each IPv4 flow stat is converted
into a CICDDoS-like feature row, classified by the trained sklearn pipeline,
logged to results/live_events.csv, and optionally mitigated with a drop rule.

Run with Ryu on Ubuntu:
    ryu-manager sdn_ryu_detector.py

Useful environment variables:
    MLDDOS_MODEL=selected_model
    MLDDOS_THRESHOLD=0.95
    MLDDOS_OBSERVE_ONLY=1
    MLDDOS_POLL_INTERVAL=3
    MLDDOS_EVENTS_CSV=/path/to/results/live_events.csv
    MLDDOS_PPS_THRESHOLD=800
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import MODELS_DIR, RESULTS_DIR

try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER
    from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
    from ryu.lib import hub
    from ryu.lib.packet import arp, ethernet, ether_types, ipv4, packet, tcp, udp
    from ryu.ofproto import ofproto_v1_3

    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False


LOGGER = logging.getLogger("ml_ddos.sdn_ryu")

DEFAULT_MODEL = "selected_model"
DEFAULT_EVENTS_CSV = RESULTS_DIR / "live_events.csv"


def safe_name(name: str) -> str:
    """Return the saved-model filename stem without importing training modules."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

LABEL_MAP = {
    0: "Benign",
    1: "LDAP Flood",
    2: "MSSQL Flood",
    3: "NetBIOS Flood",
    4: "TCP SYN Flood",
    5: "UDP Flood",
    6: "UDP-Lag Flood",
}

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


@dataclass(frozen=True)
class ControllerConfig:
    model_name: str
    model_path: Path
    events_csv: Path
    threshold: float
    observe_only: bool
    poll_interval: float
    pps_threshold: float
    drop_idle_timeout: int
    drop_hard_timeout: int
    reset_events: bool


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> ControllerConfig:
    model_name = os.environ.get("MLDDOS_MODEL", DEFAULT_MODEL)
    model_path = MODELS_DIR / f"{safe_name(model_name)}.pkl"
    if not model_path.exists() and model_name == DEFAULT_MODEL:
        fallback = MODELS_DIR / "random_forest.pkl"
        if fallback.exists():
            model_path = fallback

    return ControllerConfig(
        model_name=model_name,
        model_path=model_path,
        events_csv=Path(os.environ.get("MLDDOS_EVENTS_CSV", str(DEFAULT_EVENTS_CSV))),
        threshold=float(os.environ.get("MLDDOS_THRESHOLD", "0.95")),
        observe_only=_bool_from_env("MLDDOS_OBSERVE_ONLY", True),
        poll_interval=float(os.environ.get("MLDDOS_POLL_INTERVAL", "3")),
        pps_threshold=float(os.environ.get("MLDDOS_PPS_THRESHOLD", "800")),
        drop_idle_timeout=int(os.environ.get("MLDDOS_DROP_IDLE_TIMEOUT", "30")),
        drop_hard_timeout=int(os.environ.get("MLDDOS_DROP_HARD_TIMEOUT", "120")),
        reset_events=_bool_from_env("MLDDOS_RESET_EVENTS", False),
    )


def event_fields() -> list[str]:
    return [
        "timestamp", "src_ip", "dst_ip", "protocol",
        "fwd_packets", "bwd_packets", "prediction", "confidence",
        "is_attack", "blocked", "simulation", "model", "interface",
    ]


def ensure_events_csv(path: Path, reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not reset:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=event_fields())
        writer.writeheader()


def append_event(path: Path, event: dict[str, Any]) -> None:
    row = {field: event.get(field, "") for field in event_fields()}
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=event_fields())
        writer.writerow(row)


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x pickles to load in NumPy 1.x Ryu environments.

    Models may be trained on Windows with NumPy 2.x, whose pickle references
    numpy._core.* modules. Ryu is often pinned to older Python/NumPy versions,
    where those modules are still named numpy.core.*.
    """
    try:
        import numpy as _np

        sys.modules.setdefault("numpy._core", _np.core)
        for module_name in (
            "multiarray",
            "numeric",
            "umath",
            "fromnumeric",
            "shape_base",
            "_multiarray_umath",
        ):
            old_name = f"numpy.core.{module_name}"
            new_name = f"numpy._core.{module_name}"
            try:
                sys.modules.setdefault(new_name, importlib.import_module(old_name))
            except ImportError:
                continue
    except Exception as exc:
        LOGGER.warning("NumPy pickle compatibility shim skipped: %s", exc)


def load_model(config: ControllerConfig) -> Any | None:
    if not config.model_path.exists():
        LOGGER.error("Model not found: %s", config.model_path)
        LOGGER.error("Train first with: python main.py")
        return None
    install_numpy_pickle_compat()
    with config.model_path.open("rb") as file:
        model = pickle.load(file)
    LOGGER.info("Loaded ML model: %s", config.model_path)
    return model


def stat_match_to_dict(stat: Any) -> dict[str, Any]:
    try:
        return dict(stat.match)
    except Exception:
        return {}


def flow_stat_identity(dpid: int, stat: Any) -> tuple[Any, ...]:
    match = stat_match_to_dict(stat)
    return (
        dpid,
        match.get("in_port"),
        match.get("eth_type"),
        match.get("ipv4_src"),
        match.get("ipv4_dst"),
        match.get("ip_proto"),
        match.get("tcp_src"),
        match.get("tcp_dst"),
        match.get("udp_src"),
        match.get("udp_dst"),
    )


def protocol_name(proto: Any) -> str:
    try:
        proto_int = int(proto)
    except (TypeError, ValueError):
        return str(proto or "")
    if proto_int == 6:
        return "TCP"
    if proto_int == 17:
        return "UDP"
    if proto_int == 1:
        return "ICMP"
    return str(proto_int)


def flow_stat_to_features(stat: Any) -> pd.DataFrame:
    """Convert an OVS flow stat entry into a CICDDoS-like feature row.

    OVS provides only aggregate counters (packet_count, byte_count, duration)
    per unidirectional flow, so many CICFlowMeter bidirectional features are
    estimated or left at zero.  Protocol-aware heuristics infer flag counts
    and header lengths to give the ML model better signal.
    """
    match = stat_match_to_dict(stat)
    packet_count = float(max(getattr(stat, "packet_count", 0), 0))
    byte_count = float(max(getattr(stat, "byte_count", 0), 0))
    duration = (
        float(getattr(stat, "duration_sec", 0))
        + float(getattr(stat, "duration_nsec", 0)) / 1_000_000_000
    )
    duration_us = max(duration * 1_000_000, 1.0)
    avg_size = byte_count / packet_count if packet_count > 0 else 0.0
    packets_per_second = packet_count / max(duration, 1e-6)
    bytes_per_second = byte_count / max(duration, 1e-6)
    proto = int(match.get("ip_proto", 0) or 0)

    # Estimate packet length variance (assume uniform for single-stat flows)
    pkt_len_std = 0.0
    pkt_len_var = 0.0

    # Protocol-aware header and flag inference
    tcp_header_len = 32.0  # typical TCP header with options
    udp_header_len = 8.0
    ip_header_len = 20.0

    is_tcp = proto == 6
    is_udp = proto == 17

    fwd_header_length = packet_count * ip_header_len
    if is_tcp:
        fwd_header_length = packet_count * (ip_header_len + tcp_header_len)
    elif is_udp:
        fwd_header_length = packet_count * (ip_header_len + udp_header_len)

    # Infer flag counts from protocol and flow characteristics
    syn_flag_count = 0.0
    ack_flag_count = 0.0
    psh_flag_count = 0.0
    fin_flag_count = 0.0
    fwd_psh_flags = 0.0
    if is_tcp:
        syn_flag_count = 1.0  # at least one SYN to establish
        ack_flag_count = max(packet_count - 1.0, 0.0)
        if packet_count > 2 and byte_count > packet_count * tcp_header_len:
            psh_flag_count = max(packet_count - 2.0, 0.0)
            fwd_psh_flags = psh_flag_count

    # Segment size (payload per packet, minus headers)
    payload_per_packet = avg_size
    if is_tcp and avg_size > tcp_header_len + ip_header_len:
        payload_per_packet = avg_size - tcp_header_len - ip_header_len
    elif is_udp and avg_size > udp_header_len + ip_header_len:
        payload_per_packet = avg_size - udp_header_len - ip_header_len

    seg_size_min = ip_header_len
    if is_tcp:
        seg_size_min = ip_header_len + tcp_header_len
    elif is_udp:
        seg_size_min = ip_header_len + udp_header_len

    # Init window bytes heuristic
    init_fwd_win = 65535.0 if is_tcp else 0.0

    # Down/Up ratio
    down_up_ratio = 0.0

    row = {column: 0.0 for column in CICDDOS_FEATURE_COLUMNS}
    row.update({
        "Protocol": proto,
        "Flow Duration": duration_us,
        "Total Fwd Packets": packet_count,
        "Total Backward Packets": 0.0,
        "Fwd Packets Length Total": byte_count,
        "Bwd Packets Length Total": 0.0,
        "Fwd Packet Length Max": avg_size,
        "Fwd Packet Length Min": avg_size if packet_count else 0.0,
        "Fwd Packet Length Mean": avg_size,
        "Fwd Packet Length Std": pkt_len_std,
        "Flow Bytes/s": bytes_per_second,
        "Flow Packets/s": packets_per_second,
        "Flow IAT Mean": duration_us / max(packet_count - 1, 1),
        "Flow IAT Max": duration_us,
        "Flow IAT Min": 0.0,
        "Fwd IAT Total": duration_us,
        "Fwd IAT Mean": duration_us / max(packet_count - 1, 1),
        "Fwd IAT Max": duration_us,
        "Fwd Packets/s": packets_per_second,
        "Bwd Packets/s": 0.0,
        "Packet Length Min": avg_size if packet_count else 0.0,
        "Packet Length Max": avg_size,
        "Packet Length Mean": avg_size,
        "Packet Length Std": pkt_len_std,
        "Packet Length Variance": pkt_len_var,
        "FIN Flag Count": fin_flag_count,
        "SYN Flag Count": syn_flag_count,
        "PSH Flag Count": psh_flag_count,
        "ACK Flag Count": ack_flag_count,
        "Fwd PSH Flags": fwd_psh_flags,
        "Down/Up Ratio": down_up_ratio,
        "Avg Packet Size": avg_size,
        "Avg Fwd Segment Size": payload_per_packet,
        "Subflow Fwd Packets": packet_count,
        "Subflow Fwd Bytes": byte_count,
        "Fwd Act Data Packets": packet_count,
        "Fwd Header Length": fwd_header_length,
        "Fwd Seg Size Min": seg_size_min if packet_count else 0.0,
        "Init Fwd Win Bytes": init_fwd_win,
    })
    return pd.DataFrame([row], columns=CICDDOS_FEATURE_COLUMNS)


def class_name_from_prediction(model: Any, pred_index: int, proba: np.ndarray) -> str:
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) > pred_index:
        label = classes[pred_index]
        if isinstance(label, str):
            return label
        try:
            return LABEL_MAP.get(int(label), str(label))
        except (TypeError, ValueError):
            return str(label)
    return LABEL_MAP.get(pred_index, f"Class {pred_index}")


def predict_stat(model: Any, stat: Any, config: ControllerConfig) -> tuple[str, float]:
    features = flow_stat_to_features(stat)
    proba = np.asarray(model.predict_proba(features)[0], dtype=float)
    pred_index = int(np.argmax(proba))
    confidence = float(proba[pred_index])
    prediction = class_name_from_prediction(model, pred_index, proba)

    duration = (
        float(getattr(stat, "duration_sec", 0))
        + float(getattr(stat, "duration_nsec", 0)) / 1_000_000_000
    )
    packet_count = float(max(getattr(stat, "packet_count", 0), 0))
    pps = packet_count / max(duration, 1e-6)
    if prediction == "Benign" and config.pps_threshold > 0 and pps >= config.pps_threshold:
        prediction = "Rate-Based DDoS"
        confidence = max(confidence, 0.99)

    return prediction, confidence


def is_attack_label(label: str) -> bool:
    return label.strip().lower() not in {"benign", "normal"}


if RYU_AVAILABLE:

    class MLDDoSRyuController(app_manager.RyuApp):
        """OpenFlow 1.3 learning switch with ML-based DDoS detection."""

        OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.config = load_config()
            self.mac_to_port: dict[int, dict[str, int]] = {}
            self.datapaths: dict[int, Any] = {}
            self.last_counters: dict[tuple[Any, ...], tuple[int, int, float]] = {}
            self.last_packet_events: dict[tuple[Any, ...], float] = {}
            self.blocked_keys: set[tuple[Any, ...]] = set()
            ensure_events_csv(self.config.events_csv, reset=self.config.reset_events)
            self.model = load_model(self.config)
            self.monitor_thread = hub.spawn(self._monitor)

            mode = "OBSERVE_ONLY" if self.config.observe_only else "IPS_BLOCKING"
            LOGGER.info(
                "ML DDoS Ryu controller started [%s], threshold=%.2f, poll=%.1fs",
                mode,
                self.config.threshold,
                self.config.poll_interval,
            )
            LOGGER.info("Writing SDN events to: %s", self.config.events_csv)

        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev: Any) -> None:
            datapath = ev.msg.datapath
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            match = parser.OFPMatch()
            actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
            self.add_flow(datapath, priority=0, match=match, actions=actions)

        def add_flow(
            self,
            datapath: Any,
            priority: int,
            match: Any,
            actions: list[Any] | None,
            idle_timeout: int = 0,
            hard_timeout: int = 0,
        ) -> None:
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            instructions = []
            if actions is not None:
                instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=instructions,
                idle_timeout=idle_timeout,
                hard_timeout=hard_timeout,
            )
            datapath.send_msg(mod)

        @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
        def state_change_handler(self, ev: Any) -> None:
            datapath = ev.datapath
            if ev.state == MAIN_DISPATCHER:
                self.datapaths[datapath.id] = datapath
                LOGGER.info("Datapath connected: dpid=%s", datapath.id)
            elif ev.state == DEAD_DISPATCHER:
                self.datapaths.pop(datapath.id, None)
                LOGGER.info("Datapath disconnected: dpid=%s", datapath.id)

        @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
        def packet_in_handler(self, ev: Any) -> None:
            msg = ev.msg
            datapath = msg.datapath
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            dpid = datapath.id
            self.mac_to_port.setdefault(dpid, {})

            pkt = packet.Packet(msg.data)
            eth = pkt.get_protocol(ethernet.ethernet)
            if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
                return

            dst = eth.dst
            src = eth.src
            in_port = msg.match["in_port"]
            self.mac_to_port[dpid][src] = in_port
            out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
            actions = [parser.OFPActionOutput(out_port)]

            if out_port != ofproto.OFPP_FLOOD:
                match = self._build_forward_match(parser, in_port, eth, pkt)
                self.add_flow(
                    datapath,
                    priority=20,
                    match=match,
                    actions=actions,
                    idle_timeout=20,
                    hard_timeout=60,
                )

            self._observe_packet_in(datapath, pkt)

            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data,
            )
            datapath.send_msg(out)

        def _observe_packet_in(self, datapath: Any, pkt: Any) -> None:
            """Write low-rate PacketIn evidence while flow stats warm up."""
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt is None:
                return

            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            src_port = tcp_pkt.src_port if tcp_pkt else udp_pkt.src_port if udp_pkt else 0
            dst_port = tcp_pkt.dst_port if tcp_pkt else udp_pkt.dst_port if udp_pkt else 0
            identity = (
                datapath.id,
                ip_pkt.src,
                ip_pkt.dst,
                ip_pkt.proto,
                src_port,
                dst_port,
            )
            now = time.time()
            last = self.last_packet_events.get(identity, 0.0)
            if now - last < 2.0:
                return
            self.last_packet_events[identity] = now

            event = {
                "timestamp": datetime.now().isoformat(),
                "src_ip": ip_pkt.src,
                "dst_ip": ip_pkt.dst,
                "protocol": protocol_name(ip_pkt.proto),
                "fwd_packets": 1,
                "bwd_packets": 0,
                "prediction": "PacketIn Observed",
                "confidence": 0.0,
                "is_attack": False,
                "blocked": False,
                "simulation": self.config.observe_only,
                "model": self.config.model_name,
                "interface": f"dpid-{datapath.id}",
            }
            append_event(self.config.events_csv, event)
            LOGGER.info(
                "[PACKET_IN] %s -> %s proto=%s",
                ip_pkt.src,
                ip_pkt.dst,
                protocol_name(ip_pkt.proto),
            )

        def _build_forward_match(self, parser: Any, in_port: int, eth: Any, pkt: Any) -> Any:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt is None:
                arp_pkt = pkt.get_protocol(arp.arp)
                if arp_pkt is not None:
                    return parser.OFPMatch(
                        in_port=in_port,
                        eth_type=ether_types.ETH_TYPE_ARP,
                        eth_src=eth.src,
                        eth_dst=eth.dst,
                    )
                return parser.OFPMatch(in_port=in_port, eth_src=eth.src, eth_dst=eth.dst)

            fields: dict[str, Any] = {
                "in_port": in_port,
                "eth_type": ether_types.ETH_TYPE_IP,
                "ipv4_src": ip_pkt.src,
                "ipv4_dst": ip_pkt.dst,
                "ip_proto": ip_pkt.proto,
            }
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            if tcp_pkt is not None:
                fields["tcp_src"] = tcp_pkt.src_port
                fields["tcp_dst"] = tcp_pkt.dst_port
            elif udp_pkt is not None:
                fields["udp_src"] = udp_pkt.src_port
                fields["udp_dst"] = udp_pkt.dst_port
            return parser.OFPMatch(**fields)

        def _monitor(self) -> None:
            while True:
                if not self.datapaths:
                    LOGGER.info("Waiting for OpenFlow switch connection...")
                for datapath in list(self.datapaths.values()):
                    self._request_stats(datapath)
                hub.sleep(self.config.poll_interval)

        def _request_stats(self, datapath: Any) -> None:
            parser = datapath.ofproto_parser
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)
            LOGGER.debug("Requested flow stats from dpid=%s", datapath.id)

        @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
        def flow_stats_reply_handler(self, ev: Any) -> None:
            if self.model is None:
                return

            datapath = ev.msg.datapath
            dpid = datapath.id
            total_stats = len(ev.msg.body)
            ipv4_stats = 0
            emitted_events = 0
            for stat in ev.msg.body:
                match = stat_match_to_dict(stat)
                if int(match.get("eth_type", 0) or 0) != ether_types.ETH_TYPE_IP:
                    continue
                if "ipv4_src" not in match or "ipv4_dst" not in match:
                    continue
                if getattr(stat, "priority", 0) >= 200:
                    continue
                ipv4_stats += 1

                identity = flow_stat_identity(dpid, stat)
                packet_count = int(getattr(stat, "packet_count", 0))
                byte_count = int(getattr(stat, "byte_count", 0))
                last = self.last_counters.get(identity)
                now = time.time()
                if last is not None and packet_count <= last[0]:
                    continue
                self.last_counters[identity] = (packet_count, byte_count, now)

                try:
                    prediction, confidence = predict_stat(self.model, stat, self.config)
                except Exception as exc:
                    LOGGER.warning("Prediction failed for flow %s: %s", match, exc)
                    continue

                is_attack = is_attack_label(prediction)
                blocked = False
                if is_attack and confidence >= self.config.threshold:
                    blocked = self._maybe_block_flow(datapath, stat, identity)

                event = {
                    "timestamp": datetime.now().isoformat(),
                    "src_ip": match.get("ipv4_src", ""),
                    "dst_ip": match.get("ipv4_dst", ""),
                    "protocol": protocol_name(match.get("ip_proto")),
                    "fwd_packets": packet_count,
                    "bwd_packets": 0,
                    "prediction": prediction,
                    "confidence": round(float(confidence), 6),
                    "is_attack": is_attack,
                    "blocked": blocked,
                    "simulation": self.config.observe_only,
                    "model": self.config.model_name,
                    "interface": f"dpid-{dpid}",
                }
                append_event(self.config.events_csv, event)
                emitted_events += 1
                if is_attack:
                    LOGGER.warning(
                        "[ALERT] %s -> %s %s confidence=%.2f blocked=%s",
                        event["src_ip"],
                        event["dst_ip"],
                        prediction,
                        confidence,
                        blocked,
                    )
                else:
                    LOGGER.info(
                        "[OK] %s -> %s Benign confidence=%.2f",
                        event["src_ip"],
                        event["dst_ip"],
                        confidence,
                    )
            LOGGER.info(
                "Flow stats reply dpid=%s total=%s ipv4=%s emitted=%s",
                dpid,
                total_stats,
                ipv4_stats,
                emitted_events,
            )

        def _maybe_block_flow(self, datapath: Any, stat: Any, identity: tuple[Any, ...]) -> bool:
            if self.config.observe_only:
                return False
            if identity in self.blocked_keys:
                return True

            match_dict = stat_match_to_dict(stat)
            parser = datapath.ofproto_parser
            fields: dict[str, Any] = {
                "eth_type": ether_types.ETH_TYPE_IP,
                "ipv4_src": match_dict.get("ipv4_src"),
                "ipv4_dst": match_dict.get("ipv4_dst"),
            }
            if match_dict.get("ip_proto") is not None:
                fields["ip_proto"] = match_dict.get("ip_proto")
            for key in ("tcp_src", "tcp_dst", "udp_src", "udp_dst"):
                if match_dict.get(key) is not None:
                    fields[key] = match_dict[key]
            fields = {key: value for key, value in fields.items() if value is not None}

            drop_match = parser.OFPMatch(**fields)
            self.add_flow(
                datapath,
                priority=200,
                match=drop_match,
                actions=[],
                idle_timeout=self.config.drop_idle_timeout,
                hard_timeout=self.config.drop_hard_timeout,
            )
            self.blocked_keys.add(identity)
            LOGGER.critical("Installed SDN drop rule: %s", fields)
            return True

else:

    class MLDDoSRyuController:  # pragma: no cover - placeholder for py_compile on non-Ryu hosts.
        """Placeholder used when Ryu is not installed."""


def main() -> None:
    if RYU_AVAILABLE:
        print("This module is a Ryu app. Start it with:")
        print("  ryu-manager sdn_ryu_detector.py")
        return

    print("Ryu is not installed in this Python environment.")
    print("Use Ubuntu with Python 3.10, then install SDN requirements.")
    print("Example:")
    print("  python3.10 -m venv ryu-venv")
    print("  source ryu-venv/bin/activate")
    print("  pip install -r requirements-sdn.txt")
    sys.exit(1)


if __name__ == "__main__":
    main()
