#!/usr/bin/env python3
"""
Hierarchical / partial-pooling alternative to plain Ridge RAPM-PCV.

This fits a Gaussian MAP model (equivalent to penalized least squares) on the same
stint design used by `rapm_pcv_fit.build_rapm_mats`, but with decomposition:

  player_effect = position_group_mean + team_group_mean + player_deviation

plus optional pair (line/pair) random-effect columns for high-overlap units.

Why this helps:
- Plain Ridge shrinks each player independently toward 0.
- This model shrinks players toward *group* means first (position/team priors),
  then shrinks within-group deviations.
- Optional pair effects absorb some tightly-coupled shared signal.

Inputs:
- Stint parquet from `build_rapm_pcv.py` (e.g. `stint_pcv_targets_components_full.parquet`)
- Optional player metadata CSV/Parquet with:
    `player_id`, `position_group`, `team_group`
  (`team_group` should reflect the player's primary team in the analysis window).

Outputs:
- Parquet with hierarchical offensive/defensive coefficients and diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from rapm_pcv_fit import build_rapm_mats


ROOT = Path(__file__).resolve().parent
DEFAULT_STINTS = ROOT / "data" / "pcv" / "stint_pcv_targets_components_full.parquet"
DEFAULT_OUT = ROOT / "data" / "pcv" / "rapm_pcv_coefficients_hierarchical.parquet"


def _load_player_meta(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["player_id", "position_group", "team_group"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing --player-meta: {path}")
    if path.suffix.lower() == ".csv":
        d = pd.read_csv(path)
    else:
        d = pd.read_parquet(path)
    need = {"player_id", "position_group", "team_group"}
    miss = need.difference(d.columns)
    if miss:
        raise KeyError(f"--player-meta missing columns: {sorted(miss)}")
    d = d[list(need)].drop_duplicates("player_id", keep="last").copy()
    d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce").astype("Int64")
    d = d[d["player_id"].notna()].copy()
    d["player_id"] = d["player_id"].astype(int)
    d["position_group"] = d["position_group"].astype(str).fillna("UNK")
    d["team_group"] = d["team_group"].astype(str).fillna("UNK")
    return d


def _build_group_onehots(
    player_ids: list[int],
    meta: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[str], sparse.csr_matrix, list[str]]:
    n_p = len(player_ids)
    idx = {int(p): i for i, p in enumerate(player_ids)}
    pos_vals = sorted(meta["position_group"].dropna().astype(str).unique().tolist()) if not meta.empty else []
    team_vals = sorted(meta["team_group"].dropna().astype(str).unique().tolist()) if not meta.empty else []
    pos_map = {v: i for i, v in enumerate(pos_vals)}
    team_map = {v: i for i, v in enumerate(team_vals)}

    pr, pc = [], []
    tr, tc = [], []
    for row in meta.itertuples(index=False):
        p = int(row.player_id)
        i = idx.get(p)
        if i is None:
            continue
        pg = str(row.position_group)
        tg = str(row.team_group)
        if pg in pos_map:
            pr.append(i)
            pc.append(pos_map[pg])
        if tg in team_map:
            tr.append(i)
            tc.append(team_map[tg])

    P = sparse.csr_matrix((np.ones(len(pr)), (pr, pc)), shape=(n_p, len(pos_vals))) if pos_vals else sparse.csr_matrix((n_p, 0))
    T = sparse.csr_matrix((np.ones(len(tr)), (tr, tc)), shape=(n_p, len(team_vals))) if team_vals else sparse.csr_matrix((n_p, 0))
    return P, pos_vals, T, team_vals


def _candidate_pairs_from_stints(
    stints: pd.DataFrame,
    player_ids: list[int],
    *,
    max_pairs: int,
    min_joint_sec: float,
) -> list[tuple[int, int]]:
    target = set(int(p) for p in player_ids)
    joint: dict[tuple[int, int], float] = defaultdict(float)
    for row in stints.itertuples():
        d = float(getattr(row, "duration_sec", 0.0) or 0.0)
        if d <= 0.0:
            continue
        for col in ("home_skater_ids", "away_skater_ids"):
            side = sorted(set(int(p) for p in getattr(row, col) if int(p) in target))
            n = len(side)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    joint[(side[i], side[j])] += d
    cand = [(a, b, s) for (a, b), s in joint.items() if s >= min_joint_sec]
    cand.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in cand[: max_pairs]]


def _build_pair_design(stints: pd.DataFrame, pair_list: list[tuple[int, int]]) -> sparse.csr_matrix:
    if not pair_list:
        return sparse.csr_matrix((2 * len(stints), 0))
    p2c = {p: i for i, p in enumerate(pair_list)}
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    ridx = 0
    for row in stints.itertuples():
        hs = set(int(x) for x in row.home_skater_ids)
        aw = set(int(x) for x in row.away_skater_ids)
        for p, c in p2c.items():
            a, b = p
            if a in hs and b in hs:
                rr.append(ridx)
                cc.append(c)
                vv.append(1.0)
            if a in aw and b in aw:
                rr.append(ridx + 1)
                cc.append(c)
                vv.append(1.0)
        ridx += 2
    return sparse.csr_matrix((vv, (rr, cc)), shape=(2 * len(stints), len(pair_list)))


def _solve_map(
    C: sparse.csr_matrix,
    y: np.ndarray,
    w: np.ndarray,
    penalty_diag: np.ndarray,
) -> np.ndarray:
    ws = np.sqrt(np.asarray(w, dtype=np.float64))
    Cw = C.multiply(ws[:, None])
    yw = y * ws
    A = (Cw.T @ Cw).tocsc()
    A = A + sparse.diags(np.asarray(penalty_diag, dtype=np.float64), format="csc")
    b = Cw.T @ yw
    beta = spsolve(A, b)
    return np.asarray(beta).ravel()


def _solve_map_with_prior_mean(
    C: sparse.csr_matrix,
    y: np.ndarray,
    w: np.ndarray,
    penalty_diag: np.ndarray,
    beta_prior_mean: np.ndarray,
    beta_prior_lambda: float,
    n_beta: int,
) -> np.ndarray:
    """
    Solve MAP with prior centered on beta_prior_mean for the first n_beta coefficients:
      ||W(y - Cb)||^2 + sum_j pen_j b_j^2 + beta_prior_lambda * ||beta - m||^2
    where beta are first n_beta coefficients in b.
    """
    ws = np.sqrt(np.asarray(w, dtype=np.float64))
    Cw = C.multiply(ws[:, None])
    yw = y * ws
    A = (Cw.T @ Cw).tocsc()
    pen = np.asarray(penalty_diag, dtype=np.float64).copy()
    pen[:n_beta] = pen[:n_beta] + float(beta_prior_lambda)
    A = A + sparse.diags(pen, format="csc")
    b = Cw.T @ yw
    rhs = np.asarray(b).ravel()
    rhs[:n_beta] = rhs[:n_beta] + float(beta_prior_lambda) * np.asarray(beta_prior_mean, dtype=np.float64)
    sol = spsolve(A, rhs)
    return np.asarray(sol).ravel()


def _group_index_for_players(
    player_ids: list[int],
    meta: pd.DataFrame,
    col: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Return per-player group index (or -1 if unknown) and group labels.
    """
    if meta.empty or col not in meta.columns:
        return np.full(len(player_ids), -1, dtype=int), []
    vals = sorted(meta[col].dropna().astype(str).unique().tolist())
    gmap = {v: i for i, v in enumerate(vals)}
    pm = dict(zip(meta["player_id"].astype(int), meta[col].astype(str), strict=False))
    idx = np.full(len(player_ids), -1, dtype=int)
    for i, p in enumerate(player_ids):
        g = pm.get(int(p))
        if g in gmap:
            idx[i] = gmap[g]
    return idx, vals


def _update_group_means(
    beta: np.ndarray,
    pos_idx: np.ndarray,
    team_idx: np.ndarray,
    mu_pos: np.ndarray,
    mu_team: np.ndarray,
    *,
    lambda_player_dev: float,
    lambda_position: float,
    lambda_team: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One EB-style update for position/team prior means from current player beta.
    """
    if mu_pos.size:
        for g in range(mu_pos.size):
            m = pos_idx == g
            if not np.any(m):
                continue
            n = int(np.sum(m))
            # beta ~= mu_pos[g] + mu_team[t(i)] + dev
            resid = beta[m] - np.where(team_idx[m] >= 0, mu_team[team_idx[m]], 0.0)
            mu_pos[g] = (lambda_player_dev * float(np.sum(resid))) / (lambda_position + lambda_player_dev * n)
    if mu_team.size:
        for g in range(mu_team.size):
            m = team_idx == g
            if not np.any(m):
                continue
            n = int(np.sum(m))
            resid = beta[m] - np.where(pos_idx[m] >= 0, mu_pos[pos_idx[m]], 0.0)
            mu_team[g] = (lambda_player_dev * float(np.sum(resid))) / (lambda_team + lambda_player_dev * n)
    return mu_pos, mu_team


def fit_hierarchical_map(
    stints: pd.DataFrame,
    player_meta: pd.DataFrame,
    *,
    lambda_player_dev: float = 25000.0,
    lambda_position: float = 10000.0,
    lambda_team: float = 10000.0,
    lambda_pair: float = 12000.0,
    max_pair_effects: int = 0,
    pair_min_joint_sec: float = 6000.0,
    n_iter_prior: int = 4,
) -> pd.DataFrame:
    pids = sorted(
        set(int(p) for a in stints["home_skater_ids"] for p in a).union(set(int(p) for b in stints["away_skater_ids"] for p in b))
    )
    mats = build_rapm_mats(stints, all_player_ids=pids, sparse_mats=True)
    X = mats["X_off"].tocsr()
    Xd = mats["X_def"].tocsr()
    y_off = np.asarray(mats["y_off"], dtype=np.float64)
    y_def = np.asarray(mats["y_def"], dtype=np.float64)
    w = np.asarray(mats["w"], dtype=np.float64)

    n_p = len(pids)
    Xp = X[:, :n_p].tocsr()
    Xph = X[:, n_p : n_p + 1].tocsr()  # home indicator fixed effect
    Xpd = Xd[:, :n_p].tocsr()
    Xpdh = Xd[:, n_p : n_p + 1].tocsr()

    pos_idx, pos_names = _group_index_for_players(pids, player_meta, "position_group")
    team_idx, team_names = _group_index_for_players(pids, player_meta, "team_group")

    pair_list = _candidate_pairs_from_stints(
        stints,
        pids,
        max_pairs=int(max_pair_effects or 0),
        min_joint_sec=float(pair_min_joint_sec),
    )
    X_pair = _build_pair_design(stints, pair_list) if pair_list else sparse.csr_matrix((2 * len(stints), 0))
    Xd_pair = X_pair

    C_off = sparse.hstack([Xp, Xph, X_pair], format="csr")
    C_def = sparse.hstack([Xpd, Xpdh, Xd_pair], format="csr")

    n_pair = X_pair.shape[1]

    pen = np.concatenate(
        [
            np.full(n_p, float(lambda_player_dev)),
            np.array([0.0]),  # home fixed effect
            np.full(n_pair, float(lambda_pair)),
        ]
    )
    # Empirical-Bayes style alternating updates for prior means.
    mu_pos_o = np.zeros(len(pos_names), dtype=np.float64)
    mu_team_o = np.zeros(len(team_names), dtype=np.float64)
    mu_pos_d = np.zeros(len(pos_names), dtype=np.float64)
    mu_team_d = np.zeros(len(team_names), dtype=np.float64)
    beta_o = np.zeros(n_p, dtype=np.float64)
    beta_d = np.zeros(n_p, dtype=np.float64)

    for _ in range(max(1, int(n_iter_prior))):
        m_o = np.where(pos_idx >= 0, mu_pos_o[pos_idx], 0.0) + np.where(team_idx >= 0, mu_team_o[team_idx], 0.0)
        m_d = np.where(pos_idx >= 0, mu_pos_d[pos_idx], 0.0) + np.where(team_idx >= 0, mu_team_d[team_idx], 0.0)
        so = _solve_map_with_prior_mean(
            C_off, y_off, w, pen, m_o, float(lambda_player_dev), n_p
        )
        sd = _solve_map_with_prior_mean(
            C_def, y_def, w, pen, m_d, float(lambda_player_dev), n_p
        )
        beta_o = so[:n_p]
        beta_d = sd[:n_p]
        mu_pos_o, mu_team_o = _update_group_means(
            beta_o, pos_idx, team_idx, mu_pos_o, mu_team_o,
            lambda_player_dev=float(lambda_player_dev),
            lambda_position=float(lambda_position),
            lambda_team=float(lambda_team),
        )
        mu_pos_d, mu_team_d = _update_group_means(
            beta_d, pos_idx, team_idx, mu_pos_d, mu_team_d,
            lambda_player_dev=float(lambda_player_dev),
            lambda_position=float(lambda_position),
            lambda_team=float(lambda_team),
        )

    prior_o = np.where(pos_idx >= 0, mu_pos_o[pos_idx], 0.0) + np.where(team_idx >= 0, mu_team_o[team_idx], 0.0)
    prior_d = np.where(pos_idx >= 0, mu_pos_d[pos_idx], 0.0) + np.where(team_idx >= 0, mu_team_d[team_idx], 0.0)
    dev_o = beta_o - prior_o
    dev_d = beta_d - prior_d
    off = beta_o
    deff = beta_d

    out = pd.DataFrame(
        {
            "player_id": pids,
            "offensive_hier_rapm_pcv": off,
            "defensive_hier_rapm_pcv": deff,
            "hier_player_dev_off": dev_o,
            "hier_player_dev_def": dev_d,
        }
    )
    out["lambda_player_dev"] = float(lambda_player_dev)
    out["lambda_position"] = float(lambda_position)
    out["lambda_team"] = float(lambda_team)
    out["lambda_pair"] = float(lambda_pair)
    out["n_prior_iter"] = int(max(1, int(n_iter_prior)))
    out["n_pair_effects"] = int(n_pair)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stints", type=Path, default=DEFAULT_STINTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--player-meta",
        type=Path,
        default=None,
        help="Optional parquet/csv with player_id, position_group, team_group for hierarchical priors.",
    )
    ap.add_argument("--lambda-player-dev", type=float, default=60000.0)
    ap.add_argument("--lambda-position", type=float, default=15000.0)
    ap.add_argument("--lambda-team", type=float, default=15000.0)
    ap.add_argument("--lambda-pair", type=float, default=12000.0)
    ap.add_argument("--max-pair-effects", type=int, default=0)
    ap.add_argument("--pair-min-joint-sec", type=float, default=6000.0)
    ap.add_argument("--n-iter-prior", type=int, default=5)
    ap.add_argument("--attribution", choices=("possession_split", "xt_delta"), default="possession_split")
    args = ap.parse_args()

    if not args.stints.is_file():
        raise SystemExit(f"Missing {args.stints}")
    st = pd.read_parquet(args.stints)
    if "pcv_attribution" in st.columns:
        st = st[st["pcv_attribution"].astype(str) == str(args.attribution)].copy()
    if st.empty:
        raise SystemExit("No stint rows after attribution filter.")
    meta = _load_player_meta(args.player_meta)

    out = fit_hierarchical_map(
        st,
        meta,
        lambda_player_dev=float(args.lambda_player_dev),
        lambda_position=float(args.lambda_position),
        lambda_team=float(args.lambda_team),
        lambda_pair=float(args.lambda_pair),
        max_pair_effects=int(args.max_pair_effects),
        pair_min_joint_sec=float(args.pair_min_joint_sec),
        n_iter_prior=int(args.n_iter_prior),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  n={len(out)}")


if __name__ == "__main__":
    main()
