"""Forward slot (C/W) resolution shared by card build and weight tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd

# EH RAPM often labels wingers as C; demote when 3Y TOI-weighted faceoffs are wing-low.
CENTER_FOW_DEMOTE_THRESHOLD = 6.6


def forward_slot(
    role: str,
    eh_position: str | None = None,
    *,
    fow_per60: float | None = None,
) -> str:
    """C / W / D for overall weighting; EH L/R -> W, EH C -> C unless low faceoff volume."""
    if str(role).upper().strip() == "D":
        return "D"
    pos = str(eh_position or "").upper().strip()
    slot = "C" if pos in {"C", "C/L", "C/R"} else "W"
    if slot == "C" and fow_per60 is not None and np.isfinite(fow_per60):
        if float(fow_per60) < float(CENTER_FOW_DEMOTE_THRESHOLD):
            return "W"
    return slot


def player_weighted_fow_per60(
    df: pd.DataFrame,
    *,
    player_col: str = "player_id",
    fow_col: str = "eh_std_FOW/60",
    toi_col: str = "eh_rapm_TOI",
) -> pd.Series:
    """TOI-weighted mean faceoffs/60 across rows (e.g. 3Y window)."""
    if fow_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    fow = pd.to_numeric(df[fow_col], errors="coerce")
    if toi_col in df.columns:
        toi = pd.to_numeric(df[toi_col], errors="coerce").fillna(1.0)
    else:
        toi = pd.Series(1.0, index=df.index, dtype=float)
    tmp = df.assign(_fow=fow, _toi=toi, _wfow=fow * toi)
    agg = tmp.groupby(player_col, dropna=False).agg(_wfow=("_wfow", "sum"), _toi=("_toi", "sum"))
    out = agg["_wfow"] / agg["_toi"].replace(0.0, np.nan)
    return df[player_col].map(out)


def assign_forward_slot_column(
    df: pd.DataFrame,
    *,
    role_col: str = "position_group",
    eh_col: str = "eh_rapm_Position",
    fow_col: str = "eh_std_FOW/60",
    toi_col: str = "eh_rapm_TOI",
    player_col: str = "player_id",
    use_player_weighted_fow: bool = True,
) -> pd.Series:
    """Return forward_slot series aligned to df.index."""
    role = df.get(role_col, pd.Series("F", index=df.index)).astype(str)
    eh = df.get(eh_col, pd.Series("", index=df.index)).astype(str)
    if use_player_weighted_fow and player_col in df.columns:
        fow_s = player_weighted_fow_per60(df, player_col=player_col, fow_col=fow_col, toi_col=toi_col)
    elif fow_col in df.columns:
        fow_s = pd.to_numeric(df[fow_col], errors="coerce")
    else:
        fow_s = pd.Series(np.nan, index=df.index, dtype=float)
    return pd.Series(
        [
            forward_slot(r, e, fow_per60=float(f) if pd.notna(f) else None)
            for r, e, f in zip(role, eh, fow_s)
        ],
        index=df.index,
        dtype=str,
    )
