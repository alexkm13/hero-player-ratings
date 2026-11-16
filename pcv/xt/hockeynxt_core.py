"""
Expected Threat (xT) — logic adapted from HockeynxT
(https://github.com/adriancm93/HockeynxT, HockeynxT/xTHockey.py) and socceraction, which
implements a Markov / value-iteration form of the model described by Karun Singh
(https://karun.in/blog/expected-threat.html).

- Default **10×8** cells (l=10, w=8); `l` = divisions along 200', `w` along 85'. Upstream used 16×8.
- `ExpectedThreat` Bellman update uses a matrix form instead of 4 nested loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def nhl_to_rink_200_85(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Attacking-frame NHL (center ice origin) to [0,200] x [0,85] (neutral zone faceoff is ~100,42.5)."""
    return (x + 100.0).clip(0, 200), (y + 42.5).clip(0, 85)


def _get_cell_indexes(
    x: pd.Series, y: pd.Series, l: int, w: int
) -> Tuple[pd.Series, pd.Series]:
    xmin, ymin = 0.0, 0.0
    xi = ((x - xmin) / 200.0 * l).astype(float)
    yj = ((y - ymin) / 85.0 * w).astype(float)
    xi = xi.fillna(0).astype(int).clip(0, l - 1)
    yj = yj.fillna(0).astype(int).clip(0, w - 1)
    return xi, yj


def _get_flat_indexes(x: pd.Series, y: pd.Series, l: int, w: int) -> pd.Series:
    xi, yj = _get_cell_indexes(x, y, l, w)
    return l * (w - 1 - yj) + xi


def _count(x: pd.Series, y: pd.Series, l: int, w: int) -> np.ndarray:
    ok = x.notna() & y.notna()
    if not ok.any():
        return np.zeros((w, l), dtype=np.float64)
    flat = _get_flat_indexes(x[ok], y[ok], l, w)
    vc = flat.value_counts(sort=False)
    vec = np.zeros(w * l, dtype=np.float64)
    for i, c in zip(vc.index, vc.values):
        vec[int(i)] = float(c)
    return vec.reshape((w, l))


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def scoring_prob_shots(
    actions: pd.DataFrame,
    l: int,
    w: int,
) -> np.ndarray:
    """P(goal|shot) per cell from `Shot` rows with is_goal or success==1 for goals."""
    shots = actions[actions["type_name"] == "Shot"]
    if shots.empty:
        return np.zeros((w, l), dtype=np.float64)
    if "is_goal" in shots.columns:
        goals = shots[shots["is_goal"].astype(int) == 1]
    else:
        goals = shots[shots["success"] == 1.0]
    return _safe_divide(
        _count(goals["start_x"], goals["start_y"], l, w),
        _count(shots["start_x"], shots["start_y"], l, w),
    )


def get_move_actions(actions: pd.DataFrame) -> pd.DataFrame:
    a = actions[actions["success"].notna() & actions["end_x"].notna()].copy()
    m = a["type_name"].isin(["Zone Entry", "Pass", "Move"])
    return a[m].copy()


def action_prob(
    actions: pd.DataFrame,
    l: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = actions[actions["success"].notna() & actions["end_x"].notna()].copy()
    move = get_move_actions(a)
    shot = a[a["type_name"] == "Shot"]
    move_success = move[move["success"] == 1.0]
    movem_all = _count(move["start_x"], move["start_y"], l, w)
    movem_success = _count(move_success["start_x"], move_success["start_y"], l, w)
    shotm = _count(shot["start_x"], shot["start_y"], l, w)
    tot = movem_all + shotm
    # Continuation value only propagates through successful moves.
    return _safe_divide(shotm, tot + 1e-9), _safe_divide(movem_success, tot + 1e-9), tot


def move_transition_matrix(actions: pd.DataFrame, l: int, w: int) -> np.ndarray:
    move_actions = get_move_actions(actions)
    s = w * l
    if move_actions.empty:
        return np.eye(s, dtype=np.float64)
    X = pd.DataFrame(
        {
            "start_cell": _get_flat_indexes(
                move_actions["start_x"], move_actions["start_y"], l, w
            ).to_numpy(int, copy=False),
            "end_cell": _get_flat_indexes(
                move_actions["end_x"], move_actions["end_y"], l, w
            ).to_numpy(int, copy=False),
            "success": move_actions["success"].to_numpy(),
        }
    )
    T = np.zeros((s, s), dtype=np.float64)
    for i in range(s):
        m = (X["start_cell"] == i) & (X["success"] == 1.0)
        if not m.any():
            continue
        ends = X.loc[m, "end_cell"]
        for j_cell, c in ends.value_counts().items():
            T[i, int(j_cell)] = float(c) / float(m.sum())
    return T


@dataclass
class XTFit:
    l: int
    w: int
    xT: np.ndarray
    shot_prob: np.ndarray
    move_prob: np.ndarray
    p_goal_given_shot: np.ndarray
    transition: np.ndarray

    def xT_of_state(self, x_bin: int, y_bin: int) -> float:
        """
        `xT(state)` for discrete state (x_bin, y_bin) with 0 <= x_bin < l, 0 <= y_bin < w
        in **HockeynxT** cell indices (x along 200', y across 85' from one goal line).
        """
        if not (0 <= x_bin < self.l and 0 <= y_bin < self.w):
            return float("nan")
        return float(self.xT[self.w - 1 - y_bin, x_bin])

    def xT_nhl_feet(
        self,
        x_atk: float,
        y_atk: float,
    ) -> float:
        """NHL feet (attacking frame) — converts to 200×85 then bins, then returns xT."""
        s = pd.Series([x_atk + 100.0], dtype=float)
        t = pd.Series([y_atk + 42.5], dtype=float)
        xi, yj = _get_cell_indexes(s, t, self.l, self.w)
        return self.xT_of_state(int(xi.iloc[0]), int(yj.iloc[0]))

    def xG_nhl_feet(
        self,
        x_atk: float,
        y_atk: float,
    ) -> float:
        """
        Empirical P(goal|shot) for the binned cell (same as `p_goal_given_shot` from the fit),
        a simple xG proxy. Shot events only.
        """
        s = pd.Series([x_atk + 100.0], dtype=float)
        t = pd.Series([y_atk + 42.5], dtype=float)
        xi, yj = _get_cell_indexes(s, t, self.l, self.w)
        if pd.isna(xi.iloc[0]) or pd.isna(yj.iloc[0]):
            return float("nan")
        return float(
            self.p_goal_given_shot[
                int(self.w - 1 - int(yj.iloc[0])),
                int(xi.iloc[0]),
            ]
        )

    def xT_nhl_series(
        self,
        x_atk: pd.Series,
        y_atk: pd.Series,
    ) -> np.ndarray:
        """Vectorized xT at each row; NaN where x/y missing."""
        s = (x_atk.astype(float) + 100.0).clip(0, 200)
        t = (y_atk.astype(float) + 42.5).clip(0, 85)
        xi, yj = _get_cell_indexes(s, t, self.l, self.w)
        ok = x_atk.notna() & y_atk.notna()
        out = np.full(len(s), np.nan, dtype=np.float64)
        xb = xi[ok].astype(int, copy=False)
        yb = yj[ok].astype(int, copy=False)
        out[ok.to_numpy()] = self.xT[self.w - 1 - yb, xb]
        return out

    def xG_nhl_series(
        self,
        x_atk: pd.Series,
        y_atk: pd.Series,
    ) -> np.ndarray:
        """Vectorized P(goal|shot) per cell (xG proxy)."""
        s = (x_atk.astype(float) + 100.0).clip(0, 200)
        t = (y_atk.astype(float) + 42.5).clip(0, 85)
        xi, yj = _get_cell_indexes(s, t, self.l, self.w)
        ok = x_atk.notna() & y_atk.notna()
        out = np.full(len(s), np.nan, dtype=np.float64)
        xb = xi[ok].astype(int, copy=False)
        yb = yj[ok].astype(int, copy=False)
        out[ok.to_numpy()] = self.p_goal_given_shot[
            self.w - 1 - yb, xb
        ]
        return out


def _cell_flat(x: int, y: int, l: int, w: int) -> int:
    return l * (w - 1 - y) + x


def _solve_bellman(
    p_scoring: np.ndarray,
    p_shot: np.ndarray,
    p_move: np.ndarray,
    T: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    w, l = p_scoring.shape
    s = w * l
    gs = p_scoring * p_shot
    M = np.zeros((s, s), dtype=np.float64)
    for yy in range(w):
        for x in range(l):
            i = _cell_flat(x, yy, l, w)
            M[i, :] = p_move[yy, x] * T[i, :]
    g_vec = np.zeros(s, dtype=np.float64)
    for yy in range(w):
        for x in range(l):
            g_vec[_cell_flat(x, yy, l, w)] = gs[yy, x]
    np.nan_to_num(g_vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(M, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    xT_vec = np.zeros(s, dtype=np.float64)
    diff, it = 1.0, 0
    while diff > eps and it < 10_000:
        # Some BLAS builds emit spurious over/invalid warnings on matmul; values stay finite.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            t = M @ xT_vec
        newv = g_vec + t
        np.clip(newv, 0.0, 1.0, out=newv)
        diff = float(np.max(np.abs(newv - xT_vec)))
        xT_vec = newv
        it += 1
    return xT_vec.reshape((w, l))


class ExpectedThreat:
    def __init__(self, l: int, w: int, eps: float = 1e-5) -> None:
        self.l, self.w, self.eps = l, w, eps
        self.heatmaps: List[np.ndarray] = []
        self.xT: np.ndarray = np.zeros((w, l), dtype=np.float64)
        self.p_goal_given_shot = np.zeros((w, l), dtype=np.float64)
        self.p_shot = np.zeros((w, l), dtype=np.float64)
        self.p_move = np.zeros((w, l), dtype=np.float64)
        self.transition = np.zeros((w * l, w * l), dtype=np.float64)

    def _solve(
        self,
        p_scoring: np.ndarray,
        p_shot: np.ndarray,
        p_move: np.ndarray,
        T: np.ndarray,
    ) -> None:
        self.xT = _solve_bellman(p_scoring, p_shot, p_move, T, self.eps)

    def fit(
        self,
        actions: pd.DataFrame,
        p_goal_given_shot: np.ndarray | None = None,
    ) -> "ExpectedThreat":
        l, w = self.l, self.w
        a = actions.copy()
        if p_goal_given_shot is not None and p_goal_given_shot.shape == (w, l):
            self.p_goal_given_shot = p_goal_given_shot
        else:
            self.p_goal_given_shot = scoring_prob_shots(a, l, w)
        ps, pm, _ = action_prob(a, l, w)
        for arr in (ps, pm, self.p_goal_given_shot):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        msum = ps + pm
        if (msum < 1e-9).all():
            ps[:] = 0.5
            pm[:] = 0.5
        self.p_shot, self.p_move = ps, pm
        T = move_transition_matrix(a, l, w)
        rsum = T.sum(axis=1, keepdims=True) + 1e-9
        T = T / rsum
        for i in range(w * l):
            if T[i].sum() < 1e-6:
                T[i, i] = 1.0
        self.transition = T
        self._solve(self.p_goal_given_shot, self.p_shot, self.p_move, T)
        return self

    def to_xtfit(self) -> XTFit:
        return XTFit(
            l=self.l,
            w=self.w,
            xT=self.xT.copy(),
            shot_prob=self.p_shot.copy(),
            move_prob=self.p_move.copy(),
            p_goal_given_shot=self.p_goal_given_shot.copy(),
            transition=self.transition.copy(),
        )


def fit_xT(
    events_df: pd.DataFrame,
    grid_x: int = 10,
    grid_y: int = 8,
) -> XTFit:
    m = ExpectedThreat(l=grid_x, w=grid_y)
    m.fit(events_df)
    return m.to_xtfit()
