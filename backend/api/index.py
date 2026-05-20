"""Vercel FastAPI backend for the DDoS dashboard.

This API is intentionally lightweight. Heavy tasks such as model training,
cross-validation, packet sniffing, firewall mitigation, and SHAP generation
should be run offline or on a long-running worker, not in Vercel serverless.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


APP_DIR = Path(__file__).resolve()
PROJECT_ROOT = APP_DIR.parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = PROJECT_ROOT / "audit_results"
MODELS_DIR = PROJECT_ROOT / "saved_models"

app = FastAPI(
    title="ML DDoS Backend API",
    version="1.0.0",
    description="Lightweight API for deployment health, metrics, and dashboard metadata.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    return [_coerce_numbers(row) for row in rows]


def _coerce_numbers(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            result[key] = value
            continue

        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass

        try:
            result[key] = float(value)
            continue
        except ValueError:
            result[key] = value

    return result


def _read_text(path: Path) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    return path.read_text(encoding="utf-8")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ML DDoS Backend API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "results_available": RESULTS_DIR.exists(),
        "audit_available": AUDIT_DIR.exists(),
        "models_available": MODELS_DIR.exists(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    model_files = []
    if MODELS_DIR.exists():
        model_files = sorted(path.name for path in MODELS_DIR.glob("*.pkl"))

    return {
        "models": [
            "Random Forest",
            "KNN",
            "Extra Trees",
            "MLP Classifier",
            "XGBoost",
        ],
        "model_files": model_files,
        "classes": [
            "Benign",
            "LDAP Flood",
            "MSSQL Flood",
            "NetBIOS Flood",
            "TCP SYN Flood",
            "UDP Flood",
            "UDP-Lag Flood",
        ],
    }


@app.get("/metrics/validation")
def validation_metrics() -> list[dict[str, Any]]:
    return _read_csv(RESULTS_DIR / "validation_scores.csv")


@app.get("/metrics/test")
def test_metrics() -> list[dict[str, Any]]:
    return _read_csv(RESULTS_DIR / "test_scores.csv")


@app.get("/audit/summary")
def audit_summary() -> dict[str, str]:
    return {"markdown": _read_text(AUDIT_DIR / "audit_summary.md")}


@app.get("/audit/weighted-metrics")
def audit_weighted_metrics() -> list[dict[str, Any]]:
    return _read_csv(AUDIT_DIR / "weighted_metrics_by_split.csv")


@app.get("/audit/per-class")
def audit_per_class_metrics() -> list[dict[str, Any]]:
    return _read_csv(AUDIT_DIR / "per_class_metrics.csv")


@app.get("/audit/class-distribution")
def audit_class_distribution() -> list[dict[str, Any]]:
    return _read_csv(AUDIT_DIR / "class_distribution.csv")


@app.get("/audit/leakage-checks")
def audit_leakage_checks() -> list[dict[str, Any]]:
    return _read_csv(AUDIT_DIR / "split_leakage_checks.csv")


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    return {
        "results": sorted(path.name for path in RESULTS_DIR.glob("*")) if RESULTS_DIR.exists() else [],
        "audit_results": sorted(path.name for path in AUDIT_DIR.glob("*")) if AUDIT_DIR.exists() else [],
        "saved_models": sorted(path.name for path in MODELS_DIR.glob("*")) if MODELS_DIR.exists() else [],
    }


@app.get("/openapi-json")
def openapi_json() -> dict[str, Any]:
    return json.loads(json.dumps(app.openapi()))

