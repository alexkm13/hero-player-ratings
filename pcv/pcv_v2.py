"""
PCV V2: calibrated possession expected goals.

Three heads, all in expected-goal units:

1. Shot xG — P(goal | unblocked shot process features)
2. State EPV — P(possessing team scores within 8 seconds | current state)
3. Possession PCV — P(this possession scores), from survival xG plus pre-shot EPV

This module is standalone: it does not require the Markov xT grid. Grid xG is
only a baseline inside ``fit_pcv_v2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from coords import align_period_one_orientation, to_attacking_positive_x
from pbp_order import sort_pbp_chronological
from possession import tag_possessions

SHOT_EVENTS = frozenset({"SHOT", "MISS", "GOAL"})
SHOT_LIKE = frozenset({"SHOT", "MISS", "GOAL", "BLOCK"})
NET_X = 89.0
MAX_XG_WITH_GOALIE = 0.75
MAX_XG_EMPTY_NET = 0.98


def cap_shot_xg(p: np.ndarray, opp_empty_net: np.ndarray) -> np.ndarray:
    """Keep 5v5-style chances in a public-xG range; empty nets may stay high."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    en = np.asarray(opp_empty_net, dtype=float) == 1.0
    out = p.copy()
    out[en] = np.minimum(out[en], MAX_XG_EMPTY_NET)
    out[~en] = np.minimum(out[~en], MAX_XG_WITH_GOALIE)
    return out
EPV_HORIZON_SEC = 8.0
REBOUND_SEC = 3.0
RUSH_SEC = 4.0
RUSH_ORIGIN_X = 25.0
RUSH_MIN_DIST = 40.0
SLOT_X = 70.0

DEFAULT_GBM_PARAMS: dict[str, Any] = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "max_iter": 200,
    "l2_regularization": 0.1,
    "min_samples_leaf": 80,
    "random_state": 28,
}

SHOT_NUM = [
    "shot_distance",
    "shot_angle",
    "x_atk",
    "abs_y",
    "behind_net",
    "time_since_prev",
    "dist_from_prev",
    "speed_from_prev",
    "rebound",
    "rebound_angle_change",
    "rush",
    "period",
    "seconds_elapsed",
    "score_diff",
    "is_home",
    "skaters_for",
    "skaters_against",
    "skater_diff",
    "opp_empty_net",
    "own_empty_net",
    "prev_x_atk",
    "prev_abs_y",
    "has_shot_type",
    "has_situation",
]
SHOT_CAT = ["shot_type", "prev_event"]

EPV_NUM = SHOT_NUM + [
    "is_shot_event",
    "is_block",
    "is_turnover",
    "time_in_possession",
    "poss_event_idx",
]
EPV_CAT = ["event", "shot_type", "prev_event"]

POSS_NUM = [
    "xg_surv",
    "max_xg",
    "sum_xg",
    "n_shots",
    "n_unblocked",
    "n_blocks",
    "epv_peak_preshot",
    "duration_sec",
    "n_events",
    "ozone_events",
    "slot_events",
    "any_rebound",
    "any_rush",
    "min_shot_distance",
    "mean_shot_distance",
    "max_angle",
]


def season_start_year(nhl_game_id: int | str) -> int:
    s = str(int(nhl_game_id))
    return int(s[:4])


def is_regular_season(nhl_game_id: int | str) -> bool:
    s = str(int(nhl_game_id)).zfill(10)
    return s[5] == "2"


def parse_situation_frame(code: pd.Series) -> pd.DataFrame:
    s = code.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    ok = s.str.len().eq(4)
    digits = s.where(ok, "0000")
    out = pd.DataFrame(
        {
            "away_goalie": pd.to_numeric(digits.str[0], errors="coerce"),
            "away_skaters": pd.to_numeric(digits.str[1], errors="coerce"),
            "home_skaters": pd.to_numeric(digits.str[2], errors="coerce"),
            "home_goalie": pd.to_numeric(digits.str[3], errors="coerce"),
            "has_situation": ok.astype(float),
        },
        index=code.index,
    )
    out.loc[~ok, ["away_goalie", "away_skaters", "home_skaters", "home_goalie"]] = np.nan
    return out


def parse_situation_code(code: Any) -> dict[str, float]:
    """NHL situationCode: AG AS HS HG (away goalie, away skaters, home skaters, home goalie)."""
    row = parse_situation_frame(pd.Series([code])).iloc[0]
    return {k: float(row[k]) if pd.notna(row[k]) else float("nan") for k in row.index}


def shot_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt((NET_X - x) ** 2 + (0.0 - y) ** 2)


def shot_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(y, NET_X - x))


def survival_from_probs(probs: Iterable[float]) -> float:
    p = np.asarray(list(probs), dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    p = np.clip(p, 0.0, 0.999)
    return float(1.0 - np.prod(1.0 - p))


def _prepare_game(pbp: pd.DataFrame, nhl_game_id: int) -> pd.DataFrame:
    d = pbp.copy()
    if "possession_id" not in d.columns:
        d = tag_possessions(d)
    d = sort_pbp_chronological(d, drop_tie_col=True)
    d["NHL_game_id"] = int(nhl_game_id)
    d["season_start"] = season_start_year(nhl_game_id)
    d["game_seconds"] = (
        pd.to_numeric(d["Period"], errors="coerce").fillna(0) - 1.0
    ) * 1200.0 + pd.to_numeric(d["Seconds_Elapsed"], errors="coerce").fillna(0.0)
    home = str(d["Home_Team"].iloc[0]) if "Home_Team" in d.columns else ""
    away = str(d["Away_Team"].iloc[0]) if "Away_Team" in d.columns else ""
    ev = d["Ev_Team"].astype(str)
    is_goal = d["Event"].astype(str) == "GOAL"
    home_goal = is_goal & (ev == home)
    away_goal = is_goal & (ev == away)
    d["home_score_pre"] = home_goal.cumsum().shift(1).fillna(0.0).astype(float)
    d["away_score_pre"] = away_goal.cumsum().shift(1).fillna(0.0).astype(float)
    d = align_period_one_orientation(d)
    d = to_attacking_positive_x(d)
    d["x_atk"] = pd.to_numeric(d["xC"], errors="coerce")
    d["y_atk"] = pd.to_numeric(d["yC"], errors="coerce")
    d["abs_y"] = d["y_atk"].abs()
    d["shot_distance"] = shot_distance(d["x_atk"].to_numpy(float), d["y_atk"].to_numpy(float))
    d["shot_angle"] = shot_angle(d["x_atk"].to_numpy(float), d["y_atk"].to_numpy(float))
    d["behind_net"] = (d["x_atk"] > NET_X).astype(float)
    d["is_home"] = (ev == home).astype(float)
    d["score_diff"] = np.where(d["is_home"] == 1.0, d["home_score_pre"] - d["away_score_pre"], d["away_score_pre"] - d["home_score_pre"])
    sit = d["Situation_Code"] if "Situation_Code" in d.columns else pd.Series("", index=d.index)
    parsed = parse_situation_frame(sit)
    d["away_goalie"] = parsed["away_goalie"]
    d["away_skaters"] = parsed["away_skaters"]
    d["home_skaters"] = parsed["home_skaters"]
    d["home_goalie"] = parsed["home_goalie"]
    d["has_situation"] = parsed["has_situation"]
    d["skaters_for"] = np.where(d["is_home"] == 1.0, d["home_skaters"], d["away_skaters"])
    d["skaters_against"] = np.where(d["is_home"] == 1.0, d["away_skaters"], d["home_skaters"])
    d["skaters_for"] = d["skaters_for"].fillna(5.0)
    d["skaters_against"] = d["skaters_against"].fillna(5.0)
    d["skater_diff"] = d["skaters_for"] - d["skaters_against"]
    d["opp_empty_net"] = np.where(
        d["is_home"] == 1.0, d["away_goalie"] == 0.0, d["home_goalie"] == 0.0
    ).astype(float)
    d["own_empty_net"] = np.where(
        d["is_home"] == 1.0, d["home_goalie"] == 0.0, d["away_goalie"] == 0.0
    ).astype(float)
    d["opp_empty_net"] = d["opp_empty_net"].fillna(0.0)
    d["own_empty_net"] = d["own_empty_net"].fillna(0.0)
    if "Goalie_In_Net_Id" in d.columns:
        no_g = d["Goalie_In_Net_Id"].isna() & d["Event"].isin(SHOT_EVENTS)
        d.loc[no_g, "opp_empty_net"] = 1.0
    shot_type = d["Shot_Type"] if "Shot_Type" in d.columns else ""
    d["shot_type"] = (
        pd.Series(shot_type, index=d.index).fillna("").astype(str).str.strip().str.lower().replace({"": "unknown"})
    )
    d["has_shot_type"] = (d["shot_type"] != "unknown").astype(float)
    d["event"] = d["Event"].astype(str)
    d["is_goal"] = (d["event"] == "GOAL").astype(int)
    d["is_shot_event"] = d["event"].isin(SHOT_EVENTS).astype(float)
    d["is_block"] = (d["event"] == "BLOCK").astype(float)
    d["is_turnover"] = d["event"].isin(["GIVE", "TAKE"]).astype(float)
    d["period"] = pd.to_numeric(d["Period"], errors="coerce")
    d["seconds_elapsed"] = pd.to_numeric(d["Seconds_Elapsed"], errors="coerce")
    d["state_team"] = d["Ev_Team"].astype(str)
    if "possess_after" in d.columns:
        pa = d["possess_after"]
        use_pa = (~d["event"].isin(SHOT_EVENTS | {"BLOCK"})) & pa.notna()
        d.loc[use_pa, "state_team"] = pa[use_pa].astype(str)
    return d


def _add_sequence_features(d: pd.DataFrame) -> pd.DataFrame:
    same_period = d["Period"].eq(d["Period"].shift(1))
    dt = d["seconds_elapsed"] - d["seconds_elapsed"].shift(1)
    dt = dt.where(same_period, np.nan)
    dt = dt.clip(lower=0.0)
    d["time_since_prev"] = dt
    d["prev_event"] = d["event"].shift(1).where(same_period, "none").fillna("none")
    d["prev_x_atk"] = d["x_atk"].shift(1).where(same_period, np.nan)
    d["prev_y_atk"] = d["y_atk"].shift(1).where(same_period, np.nan)
    d["prev_abs_y"] = d["prev_y_atk"].abs()
    dx = d["x_atk"] - d["prev_x_atk"]
    dy = d["y_atk"] - d["prev_y_atk"]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    d["dist_from_prev"] = dist
    d["speed_from_prev"] = dist / dt.replace(0.0, np.nan)
    d["speed_from_prev"] = d["speed_from_prev"].replace([np.inf, -np.inf], np.nan)
    prev_shotlike = d["event"].shift(1).isin(SHOT_LIKE) & same_period
    d["rebound"] = (prev_shotlike & (dt <= REBOUND_SEC) & d["event"].isin(SHOT_EVENTS)).astype(float)
    prev_ang = shot_angle(d["prev_x_atk"].to_numpy(float), d["prev_y_atk"].fillna(0.0).to_numpy(float))
    d["rebound_angle_change"] = np.where(d["rebound"] == 1.0, np.abs(d["shot_angle"] - prev_ang), 0.0)
    rush_origin = (d["prev_x_atk"] < RUSH_ORIGIN_X) & same_period
    d["rush"] = (
        rush_origin
        & (dt <= RUSH_SEC)
        & (dist >= RUSH_MIN_DIST)
        & d["event"].isin(SHOT_EVENTS)
    ).astype(float)
    poss = d["possession_id"]
    same_poss = poss.eq(poss.shift(1)) & d["NHL_game_id"].eq(d["NHL_game_id"].shift(1))
    t0 = d["game_seconds"].where(~same_poss, np.nan)
    # first event of each possession keeps its game_seconds; later inherit via ffill per group
    d["poss_start_sec"] = d.groupby(["NHL_game_id", "possession_id"], sort=False)["game_seconds"].transform("first")
    d["time_in_possession"] = (d["game_seconds"] - d["poss_start_sec"]).clip(lower=0.0)
    d["poss_event_idx"] = d.groupby(["NHL_game_id", "possession_id"], sort=False).cumcount()
    d["in_ozone"] = (d["x_atk"] >= 25.0).astype(float)
    d["in_slot"] = ((d["x_atk"] >= SLOT_X) & (d["abs_y"] <= 20.0)).astype(float)
    return d


def _add_epv_label(d: pd.DataFrame) -> pd.DataFrame:
    """P(possessing team scores within EPV_HORIZON_SEC, including this event)."""
    team = d["state_team"].astype(str) if "state_team" in d.columns else d["Ev_Team"].astype(str)
    t = d["game_seconds"].to_numpy(float)
    goal_mask = (d["event"] == "GOAL").to_numpy()
    goal_t = t[goal_mask]
    goal_team = d["Ev_Team"].astype(str).to_numpy()[goal_mask]
    label = np.zeros(len(d), dtype=int)
    for tm in pd.unique(team):
        if not tm or tm == "nan":
            continue
        gtimes = np.sort(goal_t[goal_team == tm])
        if gtimes.size == 0:
            continue
        idx = np.where(team.to_numpy() == tm)[0]
        tt = t[idx]
        j = np.searchsorted(gtimes, tt, side="right")
        has = j < gtimes.size
        nxt = np.full(tt.shape, np.nan)
        nxt[has] = gtimes[j[has]]
        ok = has & ((nxt - tt) <= EPV_HORIZON_SEC) & (nxt > tt)
        label[idx[ok]] = 1
    d["goal_in_next_8s"] = label
    return d


def enrich_game(pbp: pd.DataFrame, nhl_game_id: int) -> pd.DataFrame:
    d = _prepare_game(pbp, nhl_game_id)
    if d.empty:
        return d
    d = _add_sequence_features(d)
    d = _add_epv_label(d)
    d["shootout"] = pd.to_numeric(d["Period"], errors="coerce").fillna(0) >= 5
    return d


def _codes(series: pd.Series, mapping: dict[str, int] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    s = series.fillna("unknown").astype(str)
    if mapping is None:
        vals = sorted(set(s.unique().tolist()) | {"unknown"})
        mapping = {v: i for i, v in enumerate(vals)}
    unk = mapping.get("unknown", 0)
    return s.map(lambda x: mapping.get(x, unk)).to_numpy(float), mapping


@dataclass
class EncodedMatrix:
    X: np.ndarray
    y: np.ndarray
    names: list[str]
    cat_mask: np.ndarray
    mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    index: pd.Index | None = None


def build_shot_matrix(
    events: pd.DataFrame,
    *,
    mappings: dict[str, dict[str, int]] | None = None,
) -> EncodedMatrix:
    m = events.loc[~events["shootout"] & events["event"].isin(SHOT_EVENTS) & events["x_atk"].notna()].copy()
    mappings = dict(mappings or {})
    cols: list[np.ndarray] = []
    names: list[str] = []
    cat_flags: list[bool] = []
    for c in SHOT_NUM:
        cols.append(pd.to_numeric(m[c], errors="coerce").to_numpy(float))
        names.append(c)
        cat_flags.append(False)
    for c in SHOT_CAT:
        arr, mp = _codes(m[c], mappings.get(c))
        mappings[c] = mp
        cols.append(arr)
        names.append(c)
        cat_flags.append(True)
    X = np.column_stack(cols) if cols else np.zeros((len(m), 0))
    y = m["is_goal"].to_numpy(int)
    return EncodedMatrix(X=X, y=y, names=names, cat_mask=np.asarray(cat_flags), mappings=mappings, index=m.index)


def build_epv_matrix(
    events: pd.DataFrame,
    *,
    mappings: dict[str, dict[str, int]] | None = None,
) -> EncodedMatrix:
    m = events.loc[~events["shootout"] & events["x_atk"].notna() & events["event"].isin(
        {"SHOT", "MISS", "BLOCK", "GIVE", "TAKE", "HIT", "FAC"}
    )].copy()
    mappings = dict(mappings or {})
    cols: list[np.ndarray] = []
    names: list[str] = []
    cat_flags: list[bool] = []
    for c in EPV_NUM:
        cols.append(pd.to_numeric(m[c], errors="coerce").to_numpy(float))
        names.append(c)
        cat_flags.append(False)
    for c in EPV_CAT:
        arr, mp = _codes(m[c], mappings.get(c))
        mappings[c] = mp
        cols.append(arr)
        names.append(c)
        cat_flags.append(True)
    X = np.column_stack(cols)
    y = m["goal_in_next_8s"].to_numpy(int)
    return EncodedMatrix(X=X, y=y, names=names, cat_mask=np.asarray(cat_flags), mappings=mappings, index=m.index)


def assign_shot_xg(events: pd.DataFrame, xg: pd.Series) -> pd.DataFrame:
    d = events.copy()
    d["xg"] = np.nan
    d.loc[xg.index, "xg"] = xg.to_numpy(float)
    return d


def assign_epv(events: pd.DataFrame, epv: pd.Series) -> pd.DataFrame:
    d = events.copy()
    d["epv"] = np.nan
    d.loc[epv.index, "epv"] = epv.to_numpy(float)
    return d


def possession_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    d = events.copy()
    if "shootout" in d.columns:
        d = d.loc[~d["shootout"]].copy()
    if d.empty:
        return pd.DataFrame()
    d["_xg_shot"] = np.where(d["event"].isin(SHOT_EVENTS), pd.to_numeric(d.get("xg", np.nan), errors="coerce"), np.nan)
    d["_epv"] = pd.to_numeric(d.get("epv", np.nan), errors="coerce")
    d["_epv_pre"] = np.where(d["event"] != "GOAL", d["_epv"], np.nan)
    d["_dist_shot"] = np.where(d["event"].isin(SHOT_EVENTS), d["shot_distance"], np.nan)
    d["_is_shot"] = d["event"].isin(SHOT_EVENTS).astype(float)
    d["_is_block"] = (d["event"] == "BLOCK").astype(float)
    d["_is_goal"] = (d["event"] == "GOAL").astype(float)
    # Administrative events (PSTR/STOP/etc.) can share the following play's
    # possession_id while carrying an empty Ev_Team.  Pandas treats an empty
    # string as a real value, so grouping on raw Ev_Team assigned roughly five
    # percent of historical possessions to no team.  Use the inferred
    # post-event state team and convert blank tokens to missing so groupby
    # ``first`` selects the first actual owner.
    state_team = d["state_team"] if "state_team" in d.columns else d["Ev_Team"]
    d["_poss_team"] = state_team.astype("string").str.strip().replace("", pd.NA)
    ev_team = d["Ev_Team"].astype("string").str.strip().replace("", pd.NA)
    d["_poss_team"] = d["_poss_team"].fillna(ev_team)
    gcols = ["NHL_game_id", "possession_id"]

    def _surv(s: pd.Series) -> float:
        return survival_from_probs(s.to_numpy(float))

    named = {
        "poss_team": ("_poss_team", "first"),
        "season_start": ("season_start", "first"),
        "xg_surv": ("_xg_shot", _surv),
        "max_xg": ("_xg_shot", "max"),
        "sum_xg": ("_xg_shot", "sum"),
        "n_shots": ("_is_shot", "sum"),
        "n_unblocked": ("_is_shot", "sum"),
        "n_blocks": ("_is_block", "sum"),
        "epv_peak": ("_epv", "max"),
        "epv_peak_preshot": ("_epv_pre", "max"),
        "epv_mean": ("_epv", "mean"),
        "duration_sec": ("time_in_possession", "max"),
        "n_events": ("event", "size"),
        "ozone_events": ("in_ozone", "sum"),
        "slot_events": ("in_slot", "sum"),
        "any_rebound": ("rebound", "max"),
        "any_rush": ("rush", "max"),
        "min_shot_distance": ("_dist_shot", "min"),
        "mean_shot_distance": ("_dist_shot", "mean"),
        "max_angle": ("shot_angle", "max"),
        "scored": ("_is_goal", "max"),
    }
    if "Home_Team" in d.columns:
        named["home_team"] = ("Home_Team", "first")
    if "Away_Team" in d.columns:
        named["away_team"] = ("Away_Team", "first")
    if "Game_Id" in d.columns:
        named["Game_Id"] = ("Game_Id", "first")
    out = d.groupby(gcols, sort=False).agg(**named).reset_index()
    out["n_shots"] = out["n_shots"].astype(int)
    out["n_unblocked"] = out["n_unblocked"].astype(int)
    out["scored"] = out["scored"].fillna(0).astype(int)
    for c in ("max_xg", "sum_xg", "epv_peak", "epv_peak_preshot", "epv_mean", "xg_surv"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def build_poss_matrix(
    poss: pd.DataFrame,
    *,
    mappings: dict[str, dict[str, int]] | None = None,
) -> EncodedMatrix:
    _ = mappings
    m = poss.copy()
    cols = [pd.to_numeric(m[c], errors="coerce").to_numpy(float) for c in POSS_NUM]
    X = np.column_stack(cols)
    y = m["scored"].to_numpy(int)
    return EncodedMatrix(
        X=X,
        y=y,
        names=list(POSS_NUM),
        cat_mask=np.zeros(len(POSS_NUM), dtype=bool),
        mappings={},
        index=m.index,
    )


def geometry_matrix(events: pd.DataFrame) -> EncodedMatrix:
    m = events.loc[~events["shootout"] & events["event"].isin(SHOT_EVENTS) & events["x_atk"].notna()].copy()
    X = np.column_stack(
        [
            pd.to_numeric(m["shot_distance"], errors="coerce").to_numpy(float),
            pd.to_numeric(m["shot_angle"], errors="coerce").to_numpy(float),
        ]
    )
    y = m["is_goal"].to_numpy(int)
    return EncodedMatrix(
        X=X,
        y=y,
        names=["shot_distance", "shot_angle"],
        cat_mask=np.zeros(2, dtype=bool),
        index=m.index,
    )


def _predict_pack(model, X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if np.isinf(values).any():
        raise ValueError("PCV V2 prediction matrix contains infinite values")
    # This host's linked BLAS emits false-positive matmul overflow warnings for
    # small, finite sklearn prediction matrices. Suppress only those known
    # messages, then fail closed if the returned probabilities are nonfinite.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*encountered in matmul", category=RuntimeWarning
        )
        p = np.asarray(model.predict_proba(values)[:, 1], dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("PCV V2 model returned nonfinite probabilities")
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def score_events_with_pack(events: pd.DataFrame, pack: dict[str, Any]) -> pd.DataFrame:
    """Attach out-of-sample ``xg`` and ``epv`` using a fitted model pack."""
    d = events.copy()
    shot = build_shot_matrix(d, mappings=pack.get("shot_mappings"))
    epv = build_epv_matrix(d, mappings=pack.get("epv_mappings"))
    d["xg"] = np.nan
    d["epv"] = np.nan
    if len(shot.y) and pack.get("shot_model") is not None:
        raw = _predict_pack(pack["shot_model"], shot.X)
        en = pd.to_numeric(d.loc[shot.index, "opp_empty_net"], errors="coerce").fillna(0.0).to_numpy(float)
        d.loc[shot.index, "xg"] = cap_shot_xg(raw, en)
    if len(epv.y) and pack.get("epv_model") is not None:
        d.loc[epv.index, "epv"] = _predict_pack(pack["epv_model"], epv.X)
    return d


def score_possessions_with_pack(events: pd.DataFrame, pack: dict[str, Any]) -> pd.DataFrame:
    d = score_events_with_pack(events, pack)
    poss = possession_feature_table(d)
    if poss.empty:
        return poss
    poss = poss.copy()
    if pack.get("poss_model") is not None:
        mat = build_poss_matrix(poss)
        pcv = _predict_pack(pack["poss_model"], mat.X)
    else:
        pcv = pd.to_numeric(poss["xg_surv"], errors="coerce").fillna(0.0).to_numpy(float)
        noshot = poss["n_shots"].to_numpy() <= 0
        if noshot.any():
            pcv[noshot] = np.clip(
                0.25
                * pd.to_numeric(poss.loc[noshot, "epv_peak_preshot"], errors="coerce").fillna(0.0).to_numpy(float),
                0.0,
                0.2,
            )
    poss["pcv"] = np.clip(np.asarray(pcv, dtype=float), 1e-6, 0.95)
    poss["pcv_sequence"] = poss["pcv"]
    poss["peak_pcv"] = np.fmax(poss["xg_surv"].to_numpy(float), poss["epv_peak_preshot"].to_numpy(float))
    poss["max_shot_xG"] = poss["max_xg"]
    poss["peak_xT"] = poss["epv_peak_preshot"]
    poss["shot_component"] = poss["xg_surv"]
    poss["threat_component"] = poss["epv_peak_preshot"]
    poss["no_shot_threat_component"] = np.where(poss["n_shots"] > 0, 0.0, poss["epv_peak_preshot"])
    poss["shot_exists"] = poss["n_shots"] > 0
    poss["pcv_source"] = np.where(poss["xg_surv"] >= poss["epv_peak_preshot"], "shot", "threat")
    return poss
