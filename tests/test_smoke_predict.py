"""Smoke test: load a trained model and run predict_proba with Ryu-like features.

Usage:
    python -m pytest tests/test_smoke_predict.py -v

If ``saved_models/selected_model.pkl`` does not exist the tests are skipped.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure backend source is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

from ml_ddos.sdn_ryu_detector import (
    CICDDOS_FEATURE_COLUMNS,
    LABEL_MAP,
    flow_stat_to_features,
)

MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_CANDIDATES = [
    MODELS_DIR / "selected_model.pkl",
    MODELS_DIR / "notebook_best_binary_model.pkl",
]


def _find_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def _load_model(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class _FakeFlowStat:
    """Minimal OpenFlow flow-stat stub for unit testing."""

    def __init__(self, *, packet_count, byte_count, duration_sec, duration_nsec, match, priority=10):
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.match = match
        self.priority = priority

    def __getattr__(self, name):
        if name == "match":
            return self.__dict__["match"]
        return self.__dict__.get(name, 0)


class _FakeMatch(dict):
    """Dict subclass with OXMTlv-like iteration for stat_match_to_dict."""

    def __iter__(self):
        for key, value in super().items():
            yield _FakeOXM(key, value)


class _FakeOXM:
    def __init__(self, name, value):
        self.header = name
        self.value = value

    def __repr__(self):
        return f"{self.header}={self.value}"


# ---------------------------------------------------------------------------
# Tests that always run (no model required)
# ---------------------------------------------------------------------------

class TestFeatureConstruction:
    """Validate flow_stat_to_features produces correct feature vectors."""

    def _make_tcp_stat(self, pkt=1000, byt=80_000, dur_sec=5):
        match = _FakeMatch({
            "eth_type": 0x0800,
            "ip_proto": 6,
            "ipv4_src": "10.0.0.1",
            "ipv4_dst": "10.0.0.2",
            "tcp_src": 12345,
            "tcp_dst": 80,
        })
        return _FakeFlowStat(
            packet_count=pkt,
            byte_count=byt,
            duration_sec=dur_sec,
            duration_nsec=0,
            match=match,
        )

    def _make_udp_stat(self, pkt=5000, byt=500_000, dur_sec=2):
        match = _FakeMatch({
            "eth_type": 0x0800,
            "ip_proto": 17,
            "ipv4_src": "10.0.0.3",
            "ipv4_dst": "10.0.0.4",
            "udp_src": 54321,
            "udp_dst": 53,
        })
        return _FakeFlowStat(
            packet_count=pkt,
            byte_count=byt,
            duration_sec=dur_sec,
            duration_nsec=0,
            match=match,
        )

    def test_feature_row_shape(self):
        stat = self._make_tcp_stat()
        df = flow_stat_to_features(stat)
        assert df.shape == (1, len(CICDDOS_FEATURE_COLUMNS))
        assert list(df.columns) == CICDDOS_FEATURE_COLUMNS

    def test_tcp_protocol_field(self):
        df = flow_stat_to_features(self._make_tcp_stat())
        assert df["Protocol"].iloc[0] == 6

    def test_udp_protocol_field(self):
        df = flow_stat_to_features(self._make_udp_stat())
        assert df["Protocol"].iloc[0] == 17

    def test_no_nan_or_inf(self):
        for stat in [self._make_tcp_stat(), self._make_udp_stat()]:
            df = flow_stat_to_features(stat)
            assert not df.isna().any().any(), "Feature row contains NaN"
            assert not np.isinf(df.values).any(), "Feature row contains Inf"

    def test_syn_flag_count_for_tcp(self):
        df = flow_stat_to_features(self._make_tcp_stat())
        assert df["SYN Flag Count"].iloc[0] >= 1.0

    def test_syn_flag_zero_for_udp(self):
        df = flow_stat_to_features(self._make_udp_stat())
        assert df["SYN Flag Count"].iloc[0] == 0.0

    def test_flow_duration_positive(self):
        df = flow_stat_to_features(self._make_tcp_stat(dur_sec=10))
        assert df["Flow Duration"].iloc[0] == 10 * 1_000_000

    def test_iat_fields_populated(self):
        df = flow_stat_to_features(self._make_tcp_stat(pkt=100))
        assert df["Flow IAT Mean"].iloc[0] > 0
        assert df["Fwd IAT Total"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Tests that require a trained model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    path = _find_model()
    if path is None:
        pytest.skip(
            "No trained model found. Run the training pipeline first: "
            "python -m ml_ddos.main"
        )
    return _load_model(path)


class TestModelPredict:
    """End-to-end: build Ryu feature vector → predict_proba."""

    def _make_ryu_tcp_syn_flood_features(self):
        """Simulate a TCP SYN flood: many small packets, short duration."""
        match = _FakeMatch({
            "eth_type": 0x0800,
            "ip_proto": 6,
            "ipv4_src": "192.168.1.100",
            "ipv4_dst": "10.0.0.1",
            "tcp_src": 0,
            "tcp_dst": 80,
        })
        stat = _FakeFlowStat(
            packet_count=50000,
            byte_count=50000 * 54,  # small SYN packets
            duration_sec=1,
            duration_nsec=0,
            match=match,
        )
        return flow_stat_to_features(stat)

    def _make_ryu_benign_features(self):
        """Simulate normal HTTP traffic."""
        match = _FakeMatch({
            "eth_type": 0x0800,
            "ip_proto": 6,
            "ipv4_src": "192.168.1.10",
            "ipv4_dst": "10.0.0.1",
            "tcp_src": 45678,
            "tcp_dst": 443,
        })
        stat = _FakeFlowStat(
            packet_count=150,
            byte_count=150 * 800,
            duration_sec=30,
            duration_nsec=0,
            match=match,
        )
        return flow_stat_to_features(stat)

    def _make_ryu_udp_flood_features(self):
        """Simulate a UDP flood: high packet rate, small packets."""
        match = _FakeMatch({
            "eth_type": 0x0800,
            "ip_proto": 17,
            "ipv4_src": "192.168.1.200",
            "ipv4_dst": "10.0.0.1",
            "udp_src": 12345,
            "udp_dst": 53,
        })
        stat = _FakeFlowStat(
            packet_count=100000,
            byte_count=100000 * 60,
            duration_sec=2,
            duration_nsec=0,
            match=match,
        )
        return flow_stat_to_features(stat)

    def test_predict_proba_returns_valid_shape(self, model):
        features = self._make_ryu_tcp_syn_flood_features()
        proba = model.predict_proba(features)
        assert proba.shape[0] == 1
        assert proba.shape[1] == len(LABEL_MAP)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_predict_returns_valid_label(self, model):
        features = self._make_ryu_benign_features()
        pred = model.predict(features)
        assert len(pred) == 1
        assert int(pred[0]) in LABEL_MAP

    def test_predict_proba_all_scenarios(self, model):
        scenarios = {
            "tcp_syn_flood": self._make_ryu_tcp_syn_flood_features(),
            "benign": self._make_ryu_benign_features(),
            "udp_flood": self._make_ryu_udp_flood_features(),
        }
        for name, features in scenarios.items():
            proba = model.predict_proba(features)
            pred = model.predict(features)
            label = LABEL_MAP.get(int(pred[0]), "Unknown")
            confidence = float(proba.max())
            print(f"  {name}: predicted={label}, confidence={confidence:.4f}")
            assert 0.0 <= confidence <= 1.0
