"""Build the HERO V8 three-season rating from locally supplied category scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = (
    "cat_shot_creation_z", "cat_transition_offense_z", "cat_shot_defense_z",
    "cat_transition_defense_z", "cat_finishing_z", "cat_playmaking_z",
    "cat_forecheck_z", "cat_physicality_z", "cat_skating_z",
)
RECENCY = 0.40
SHRINK_K = 400.0


def number(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def category_panel(tuned, weights, cutoff):
    eligible = tuned["model_rating_eligible"].fillna(False).astype(bool)
    panel = tuned.loc[
        eligible & tuned.season_id.isin([cutoff - 20002, cutoff - 10001, cutoff])
        & tuned.position_group.isin(["F", "D"])
    ].copy()
    if panel.duplicated(["player_id", "season_id"]).any():
        raise ValueError("Category inputs need one row per player and season.")
    eh, a3z = number(panel.eh_rapm_TOI), number(panel.a3z_toi_5v5_min)
    panel["toi"] = eh.where(eh.gt(0), a3z).fillna(0.0)
    panel = panel[panel.toi.gt(0)].copy()
    panel["season_score_raw"] = 0.0
    for category in CATEGORIES:
        role_weight = panel.position_group.map(
            {role: float(weights[f"weights_{role}"][category]) for role in ("F", "D")}
        )
        panel["season_score_raw"] += number(panel[category]).fillna(0.0) * role_weight
    panel["single_ability_raw"] = (
        panel.season_score_raw * panel.toi / (panel.toi + SHRINK_K)
    )
    panel["single_ability_pct"] = (
        panel.groupby("season_id").single_ability_raw.rank(method="average", pct=True) * 100.0
    )
    return panel


def pool_as_of(panel, cutoff):
    latest = panel.loc[panel.season_id.eq(cutoff), ["player_id", "player_name", "position_group"]]
    hist = panel.loc[
        panel.season_id.isin([cutoff - 20002, cutoff - 10001, cutoff])
        & panel.player_id.isin(latest.player_id)
    ].copy()
    hist["pool_weight"] = hist.toi * RECENCY ** ((cutoff - hist.season_id) // 10001)
    hist["weighted_score"] = hist.pool_weight * hist.season_score_raw
    totals = hist.groupby("player_id").agg(
        pool_weight=("pool_weight", "sum"), weighted_score=("weighted_score", "sum"),
        window_toi=("toi", "sum"), evidence_seasons=("season_id", "nunique")
    )
    totals["current_ability_raw"] = (
        (totals.weighted_score / totals.pool_weight)
        * (totals.window_toi / (totals.window_toi + SHRINK_K))
    )
    return latest.merge(
        totals[["current_ability_raw", "window_toi", "evidence_seasons"]],
        left_on="player_id", right_index=True, validate="one_to_one",
    )


def rank_cards(prior, card_ids=None):
    if card_ids is not None:
        wanted = set(card_ids)
        prior = prior[prior.player_id.isin(wanted)]
        missing = wanted - set(prior.player_id)
        if missing:
            raise ValueError(f"{len(missing)} requested skaters lack a prior at this cutoff.")
    out = prior.sort_values(["current_ability_raw", "player_id"], ascending=[False, True]).copy()
    if out.empty:
        raise ValueError("No eligible skaters at the requested cutoff.")
    out.insert(0, "league_rank", np.arange(1, len(out) + 1))
    out["current_ability_pct"] = out.current_ability_raw.rank(method="average", pct=True) * 100.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", type=Path, required=True, help="Tuned player-season parquet.")
    parser.add_argument("--weights", type=Path, required=True, help="V7 role weights JSON.")
    parser.add_argument("--card-list", type=Path, help="Optional CSV with the published player_id cohort.")
    parser.add_argument("--cutoff", type=int, default=20252026, help="Last completed season, e.g. 20252026.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = json.loads(args.weights.read_text())
    panel = category_panel(pd.read_parquet(args.categories), weights, args.cutoff)
    ids = pd.read_csv(args.card_list).player_id if args.card_list else None
    result = rank_cards(pool_as_of(panel, args.cutoff), ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} skater ratings to {args.output}")


if __name__ == "__main__":
    main()
