"""
Five-pillar skating skill model (Speed, Acceleration, Agility, Balance, Endurance).

Used by build_all_in_one_8_category_model.py (cat_skating) and card rendering.
EDGE supplies speed/acceleration/endurance and EDGE-only agility/balance proxies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_SKATING_PILLAR_WEIGHTS = ROOT / "data/pcv/skating_component_weights_best.json"

PILLAR_COLS = (
    "skating_pillar_speed",
    "skating_pillar_acceleration",
    "skating_pillar_agility",
    "skating_pillar_balance",
    "skating_pillar_endurance",
)

PILLAR_JSON_KEYS = ("speed", "acceleration", "agility", "balance", "endurance")

DEFAULT_PILLAR_WEIGHTS: dict[str, dict[str, float]] = {
    "F": {
        "speed": 0.14,
        "acceleration": 0.06,
        "agility": 0.50,
        "balance": 0.04,
        "endurance": 0.26,
    },
    "D": {
        "speed": 0.12,
        "acceleration": 0.18,
        "agility": 0.18,
        "balance": 0.17,
        "endurance": 0.35,
    },
}


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    raw = df.get(col)
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _first_series(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        s = _num(df, col)
        out = out.where(out.notna(), s)
    return out


def load_skating_pillar_weights(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load role weights; accepts legacy speed/burst/engine/mobility json with mapping."""
    path = path if path is not None else DEFAULT_SKATING_PILLAR_WEIGHTS
    out = {role: dict(DEFAULT_PILLAR_WEIGHTS[role]) for role in ("F", "D")}
    if not path.is_file():
        return out
    obj = json.loads(path.read_text(encoding="utf-8"))
    for role in ("F", "D"):
        src = obj.get(role, {})
        if not isinstance(src, dict):
            continue
        if all(k in src for k in PILLAR_JSON_KEYS):
            for k in PILLAR_JSON_KEYS:
                v = src.get(k)
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    out[role][k] = float(v)
        else:
            # Legacy keys: burst->acceleration, engine->endurance, mobility->agility.
            legacy_map = {
                "speed": "speed",
                "burst": "acceleration",
                "engine": "endurance",
                "mobility": "agility",
            }
            for old, new in legacy_map.items():
                v = src.get(old)
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    out[role][new] = float(v)
        s = sum(max(0.0, float(out[role][k])) for k in PILLAR_JSON_KEYS)
        if s > 0.0:
            for k in PILLAR_JSON_KEYS:
                out[role][k] = max(0.0, float(out[role][k])) / s
    return out


def add_skating_pillar_features(df: pd.DataFrame) -> None:
    """In-place: raw pillar inputs on 0–1 NHL pct scale where applicable."""
    over20 = _num(df, "edge_bursts_over20_pct")
    over22 = _num(df, "edge_bursts_over22_pct")
    b2022 = _num(df, "edge_bursts_20_to_22_pct")
    b1820 = _num(df, "edge_bursts_18_to_20_pct")

    acc_parts = [
        (over20, 0.40),
        (over22, 0.25),
        (b2022, 0.20),
        (b1820, 0.15),
    ]
    acc_num = pd.Series(0.0, index=df.index, dtype="float64")
    acc_den = pd.Series(0.0, index=df.index, dtype="float64")
    for s, w in acc_parts:
        m = s.notna()
        acc_num = acc_num + np.where(m, s * w, 0.0)
        acc_den = acc_den + np.where(m, w, 0.0)
    acceleration = pd.Series(np.nan, index=df.index, dtype="float64")
    m_acc = acc_den > 0.0
    acceleration.loc[m_acc] = acc_num.loc[m_acc] / acc_den.loc[m_acc]
    acceleration = acceleration.where(acceleration.notna(), over20)

    # Endurance: sustained output (period/game peaks), not season mileage workload.
    endurance = _first_series(
        df,
        (
            "edge_es_distance_max_period_pct",
            "edge_es_distance_max_game_pct",
            "edge_distance_max_game_pct",
        ),
    )

    df["skating_pillar_speed"] = _num(df, "edge_speed_max_pct")
    df["skating_pillar_acceleration"] = acceleration
    df["skating_pillar_endurance"] = endurance

    speed = _num(df, "edge_speed_max_pct")
    nz_time = _num(df, "edge_nz_time_pctg_pct")
    oz_time = _num(df, "edge_oz_time_ev_pctg_pct")
    dz_time = _num(df, "edge_dz_time_pctg_pct")
    distance = _first_series(df, ("edge_es_distance_per60_pct", "edge_all_distance_per60_pct"))

    # Keep skating disjoint from A3Z transition micro. These EDGE-only proxies
    # are weaker than entry/carry micro, but they prevent hidden double-counting.
    df["skating_pillar_agility"] = (
        0.45 * acceleration.where(acceleration.notna(), speed)
        + 0.35 * speed.where(speed.notna(), acceleration)
        + 0.20 * nz_time.where(nz_time.notna(), distance)
    )
    df["skating_pillar_balance"] = (
        0.45 * acceleration.where(acceleration.notna(), speed)
        + 0.30 * endurance.where(endurance.notna(), distance)
        + 0.25 * (oz_time - dz_time).where(oz_time.notna() & dz_time.notna(), nz_time)
    )


def skating_model_hybrid_spec(role: str, weights: dict[str, float]) -> list[tuple[str, float]]:
    role = str(role).upper().strip()
    w = weights if role == "F" else weights
    return [(f"skating_pillar_{k}", float(w.get(k, 0.0))) for k in PILLAR_JSON_KEYS if float(w.get(k, 0.0)) > 0.0]


def blend_skating_pillar_percentiles(
    pillar_pct: dict[str, float],
    role: str,
    weights: dict[str, dict[str, float]] | None = None,
) -> float:
    """Weighted blend of league percentiles (0–100) for card display."""
    role = str(role).upper().strip()
    wmap = weights if weights is not None else load_skating_pillar_weights()
    ws = wmap.get(role, wmap["F"])
    vals: list[float] = []
    wts: list[float] = []
    for key in PILLAR_JSON_KEYS:
        col = f"skating_pillar_{key}"
        v = pillar_pct.get(col)
        if v is None:
            v = pillar_pct.get(f"pct__{col}")
        v = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
        wt = float(ws.get(key, 0.0))
        if np.isfinite(v) and wt > 0.0:
            vals.append(v)
            wts.append(wt)
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=wts))
