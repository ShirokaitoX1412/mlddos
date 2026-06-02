"""
replay_ips.py - Offline/live replay IDS/IPS demo without a network card.

This module replays CICDDoS2019 flow rows as simulated real-time events,
runs the saved ML pipeline, and writes dashboard-compatible events to
results/live_events.csv. It is intended for demos where live packet capture
is unavailable or unreliable.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data_loader import collect_file_paths
from .models import load_model, safe_name
from .paths import DATA_DIR, MODELS_DIR, RESULTS_DIR
from .preprocessor import DDOS_CATEGORY_MAP, LABEL_HARMONIZATION_MAP, TARGET_COL


DEFAULT_MODEL = "selected_model"
DEFAULT_EVENTS_CSV = RESULTS_DIR / "live_events.csv"
DEFAULT_LIMIT = 500


EVENT_FIELDS = [
    "timestamp",
    "src_ip",
    "dst_ip",
    "protocol",
    "fwd_packets",
    "bwd_packets",
    "prediction",
    "confidence",
    "is_attack",
    "blocked",
    "simulation",
    "model",
    "interface",
    "actual_label",
    "source",
]


def _ensure_events_csv(path: Path, reset: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset or not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=EVENT_FIELDS).writeheader()


def _append_event(path: Path, event: dict) -> None:
    row = {field: event.get(field, "") for field in EVENT_FIELDS}
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EVENT_FIELDS).writerow(row)


def _fake_ip(index: int, subnet: str) -> str:
    """Generate deterministic non-routable demo IPs."""
    host = (index % 253) + 2
    return f"{subnet}.{host}"


def _first_existing(row: pd.Series, columns: Iterable[str], default=0):
    for col in columns:
        if col in row and pd.notna(row[col]):
            return row[col]
    return default


def _event_from_prediction(
    row: pd.Series,
    prediction: str,
    confidence: float,
    actual_label: str,
    index: int,
    model_name: str,
    source_name: str,
) -> dict:
    protocol = int(_first_existing(row, ["Protocol", "protocol"], 0))
    fwd_packets = int(_first_existing(row, ["Total Fwd Packets", "Fwd Packet Count", "fwd_packets"], 0))
    bwd_packets = int(_first_existing(row, ["Total Backward Packets", "Bwd Packet Count", "bwd_packets"], 0))

    return {
        "timestamp": datetime.now().isoformat(),
        "src_ip": _fake_ip(index, "10.10.1"),
        "dst_ip": _fake_ip(index * 7, "10.10.2"),
        "protocol": protocol,
        "fwd_packets": fwd_packets,
        "bwd_packets": bwd_packets,
        "prediction": prediction,
        "confidence": float(confidence),
        "is_attack": prediction != "Benign",
        "blocked": False,
        "simulation": True,
        "model": model_name,
        "interface": "offline_replay",
        "actual_label": actual_label,
        "source": source_name,
    }


def _load_source_data(source: str | None, limit: int | None, shuffle: bool, random_state: int) -> pd.DataFrame:
    if source:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Replay source not found: {path}")
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path)
        else:
            raise ValueError("Replay source must be .parquet or .csv")
        if TARGET_COL in df.columns:
            df[TARGET_COL] = _harmonize_replay_labels(df[TARGET_COL])
        if shuffle:
            df = df.sample(frac=1.0, random_state=random_state)
        return df.head(limit) if limit else df

    _, test_paths = collect_file_paths(str(DATA_DIR))
    if test_paths:
        frames = []
        remaining = limit
        for test_path in test_paths:
            chunk = pd.read_parquet(test_path)
            if shuffle:
                chunk = chunk.sample(frac=1.0, random_state=random_state)
            if remaining is not None:
                chunk = chunk.head(max(remaining, 0))
                remaining -= len(chunk)
            frames.append(chunk)
            if remaining is not None and remaining <= 0:
                break

        test_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if TARGET_COL in test_df.columns:
            test_df[TARGET_COL] = _harmonize_replay_labels(test_df[TARGET_COL])
        if shuffle:
            test_df = test_df.sample(frac=1.0, random_state=random_state)
        return test_df.head(limit) if limit else test_df

    raise FileNotFoundError(
        f"No replay source found. Put *-testing.parquet files in {DATA_DIR} "
        "or pass --source path/to/flows.parquet."
    )


def _harmonize_replay_labels(labels: pd.Series) -> pd.Series:
    """Map raw CICDDoS labels to the descriptive names used by training."""
    normalized = labels.map(LABEL_HARMONIZATION_MAP).fillna(labels)
    return normalized.map(DDOS_CATEGORY_MAP).fillna(normalized)


def _load_replay_model(model_name: str):
    try:
        return load_model(model_name, str(MODELS_DIR))
    except FileNotFoundError:
        if model_name == DEFAULT_MODEL:
            fallback = MODELS_DIR / "random_forest.pkl"
            if fallback.exists():
                print("[replay_ips] selected_model.pkl not found. Falling back to random_forest.pkl")
                with fallback.open("rb") as f:
                    return pickle.load(f)
        raise


def _label_map_from_model(model) -> dict[int, str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        classes = getattr(inner, "classes_", None)
    default = {
        0: "Benign",
        1: "LDAP Flood",
        2: "MSSQL Flood",
        3: "NetBIOS Flood",
        4: "TCP SYN Flood",
        5: "UDP Flood",
        6: "UDP-Lag Flood",
    }
    if classes is None:
        return default
    return {int(cls): default.get(int(cls), f"Class {int(cls)}") for cls in classes}


def run_replay(
    model_name: str,
    source: str | None,
    events_csv: Path,
    limit: int | None,
    speed: float,
    threshold: float,
    reset_log: bool,
    shuffle: bool,
    random_state: int,
) -> int:
    model = _load_replay_model(model_name)
    label_map = _label_map_from_model(model)
    df = _load_source_data(source, limit, shuffle, random_state)

    if df.empty:
        print("[replay_ips] No rows to replay.")
        return 0

    _ensure_events_csv(events_csv, reset_log)

    actual_labels = df[TARGET_COL].astype(str).tolist() if TARGET_COL in df.columns else [""] * len(df)
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    source_name = Path(source).name if source else "CICDDoS2019 testing parquet"
    safe_model_name = safe_name(model_name)

    print(f"[replay_ips] Replaying {len(X)} flow rows")
    print(f"[replay_ips] Model: {model_name} ({safe_model_name}.pkl)")
    print(f"[replay_ips] Events CSV: {events_csv}")
    print(f"[replay_ips] Speed: {speed}s per flow | threshold: {threshold}")

    written = 0
    for idx, (_, row) in enumerate(X.iterrows(), start=1):
        row_df = row.to_frame().T
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(row_df)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])
        else:
            pred_class = int(model.predict(row_df)[0])
            confidence = 1.0

        prediction = label_map.get(pred_class, f"Class {pred_class}")
        event = _event_from_prediction(
            row=row,
            prediction=prediction,
            confidence=confidence,
            actual_label=actual_labels[idx - 1],
            index=idx,
            model_name=model_name,
            source_name=source_name,
        )
        event["blocked"] = bool(event["is_attack"] and confidence >= threshold)

        _append_event(events_csv, event)
        written += 1

        status = "BLOCK" if event["blocked"] else ("ALERT" if event["is_attack"] else "OK")
        print(
            f"[{status}] {event['src_ip']} -> {event['dst_ip']} | "
            f"{prediction} ({confidence:.1%}) | actual={event['actual_label']}"
        )

        if speed > 0:
            time.sleep(speed)

    print(f"[replay_ips] Done. Wrote {written} events to {events_csv}")
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Replay CICDDoS flows into dashboard-compatible IPS events.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Saved model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--source", "-s", default=None, help="Optional .parquet/.csv flow source. Defaults to data/*-testing.parquet")
    parser.add_argument("--events-csv", default=str(DEFAULT_EVENTS_CSV), help=f"Output event CSV (default: {DEFAULT_EVENTS_CSV})")
    parser.add_argument("--limit", "-l", type=int, default=DEFAULT_LIMIT, help=f"Max rows to replay (default: {DEFAULT_LIMIT}; 0 = all)")
    parser.add_argument("--speed", type=float, default=0.2, help="Delay between replayed flows in seconds (default: 0.2)")
    parser.add_argument("--threshold", "-t", type=float, default=0.95, help="Simulated block threshold (default: 0.95)")
    parser.add_argument("--append", action="store_true", help="Append to existing live_events.csv instead of resetting it")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle replay rows before emitting events")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for shuffle/fake event repeatability")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = None if args.limit == 0 else args.limit
    run_replay(
        model_name=args.model,
        source=args.source,
        events_csv=Path(args.events_csv),
        limit=limit,
        speed=args.speed,
        threshold=args.threshold,
        reset_log=not args.append,
        shuffle=args.shuffle,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
