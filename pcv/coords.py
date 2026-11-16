"""
NHL PBP coordinates: xC/yC from the league (feet; center = 0; boards about ±100 in x, ±42.5 in y).

The league flips the coordinate system each period (teams change ends). To compare events
in a “period 1 orientation” frame, reflect even-numbered *regulation* periods. For OT, the
pattern continues (period 4 behaves like 2, etc.); exclude shootout rows (Period=5 and Event in
SOC) if present.

“Attacking toward +x” (single offensive direction): after period unification, flip x (and y)
so both teams’ shots at the net they attack end on the +x side. Whether “home attacks +x” in
period-1 coordinates is arena-dependent, so we infer it from period-1 shot locations: home
offensive-zone attempts (Home_Zone=Off) should have median x on the side of the net they
attack; if that median is negative, we flip home instead of away.
"""

from __future__ import annotations

import pandas as pd

SHOT_LIKE = {"SHOT", "GOAL", "MISS", "BLOCK"}


def align_period_one_orientation(
    df: pd.DataFrame,
    *,
    xcol: str = "xC",
    ycol: str = "yC",
) -> pd.DataFrame:
    """
    Reflect x,y in even standard periods (2,4,...) to match period-1 goal orientation.
    """
    out = df.copy()
    p = out["Period"].fillna(0).astype(int)
    flip = (p > 0) & (p % 2 == 0)
    for c in (xcol, ycol):
        if c in out.columns:
            v = out[c].astype(float)
            out.loc[flip, c] = -v[flip]
    return out


def _home_attacks_positive_x_p1(df: pd.DataFrame) -> bool:
    """Infer from period-1 shot-like events with coordinates (pre- or post-align; P1 same)."""
    home = df["Home_Team"].iloc[0]
    away = df["Away_Team"].iloc[0]
    m = (
        (df["Period"] == 1)
        & df["Event"].isin(SHOT_LIKE)
        & df["xC"].notna()
    )
    p1 = df.loc[m]
    h_oz = p1[(p1["Ev_Team"] == home) & (p1["Home_Zone"] == "Off")]
    a_oz = p1[(p1["Ev_Team"] == away) & (p1["Home_Zone"] == "Def")]
    if len(h_oz) >= 2:
        return bool(h_oz["xC"].astype(float).median() > 0)
    if len(a_oz) >= 2:
        # Away OZ: if home attacks +x, away shoots from x < 0
        return bool(a_oz["xC"].astype(float).median() < 0)
    return True


def to_attacking_positive_x(
    df: pd.DataFrame,
    *,
    xcol: str = "xC",
    ycol: str = "yC",
) -> pd.DataFrame:
    """
    After align_period_one_orientation, map so each Ev_Team’s offensive direction is +x.

    Negates (x,y) for away skaters when home attacks +x in period-1 data; otherwise negates
    home, so arena orientation is handled automatically.
    """
    out = df.copy()
    if out.empty:
        return out

    def _flip_one_game(g: pd.DataFrame) -> pd.DataFrame:
        z = g.copy()
        home_attacks_pos = _home_attacks_positive_x_p1(z)
        home = z["Home_Team"].iloc[0]
        is_home = z["Ev_Team"] == home
        flip_away = home_attacks_pos
        for c in (xcol, ycol):
            if c not in z.columns:
                continue
            v = z[c].astype(float)
            if flip_away:
                z.loc[~is_home, c] = -v[~is_home]
            else:
                z.loc[is_home, c] = -v[is_home]
        return z

    if "Game_Id" in out.columns and out["Game_Id"].nunique(dropna=True) > 1:
        parts = [_flip_one_game(g) for _, g in out.groupby("Game_Id", sort=False)]
        return pd.concat(parts, ignore_index=False).sort_index(kind="mergesort")
    return _flip_one_game(out)
