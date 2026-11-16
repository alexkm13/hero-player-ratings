"""Shared category lists and role-specific specs for the all-in-one model."""

from __future__ import annotations

from a3z_feature_integrity import assert_no_deprecated_a3z_features

# EH columns built in build_all_in_one_8_category_model._add_finishing_derived_features
FINISHING_DERIVED_COLS = (
    "eh_finishing_goals_above_ixg_per_60",
    "eh_finishing_sh_pct_above_xsh",
    "eh_finishing_goals_above_ixg_per_60_blended",
    "eh_finishing_sh_pct_above_xsh_blended",
    "eh_finishing_g_per_shot_above_xg_per_shot_blended",
    "eh_finishing_goals_above_ixg_per_60_shrunk",
    "eh_finishing_sh_pct_above_xsh_shrunk",
    "eh_finishing_g_per_shot_above_xg_per_shot_shrunk",
)

EXTRA_CATEGORY_SPECS_BY_ROLE: dict[str, dict[str, list[tuple[str, float]]]] = {
    "F": {
        # Isolated: A3Z forecheck/recovery micro only (no PTV, no turnover/penalty overlap).
        "Forecheck": [
            # Cycle pressure is substantially more repeatable YoY than generic
            # pressure or recovery volume; retain the latter as supporting signals.
            ("a3z_forecheck_pressures_per_60", 0.30),
            ("a3z_cycle_forecheck_events_per_60", 0.55),
            ("a3z_recoveries_per_60", 0.15),
        ],
        # Physicality: hits and blocks only.
        "Physicality": [
            ("eh_std_iHF/60", 0.65),
            ("eh_std_iBLK/60", 0.35),
        ],
    },
    "D": {
        "Forecheck": [
            ("a3z_forecheck_pressures_per_60", 0.25),
            ("a3z_cycle_forecheck_events_per_60", 0.65),
            ("a3z_recoveries_per_60", 0.10),
        ],
        # Physicality: hits and blocks only (D get more credit for blocks).
        "Physicality": [
            ("eh_std_iHF/60", 0.45),
            ("eh_std_iBLK/60", 0.40),
            ("eh_std_sh_iBLK/60", 0.15),
        ],
    },
}

COMPONENT_FEATURE_CONTRACTS: dict[str, frozenset[str]] = {
    "cat_shot_creation": frozenset(
        {
            "rapm_hybrid_xGF/60",
            "eh_std_ixG/60",
            "eh_std_iCF/60",
            "a3z_chances_per_60",
            "a3z_carries_with_chances_per_60",
            "a3z_shots_per_60",
            "ptv_coalesced_rapm_shot_threat",
            # Feature-table stem (not the rapm_ alias); TO owns the alias copy.
            "ptv_coalesced_possession_offense",
        }
    ),
    "cat_transition_offense": frozenset(
        {
            "a3z_entries_per_60",
            "a3z_carries_per_60",
            "a3z_entries_with_passing_play_per_60",
            "a3z_controlled_exits_per_60",
            "a3z_carried_exits_per_60",
            "a3z_passed_exits_per_60",
            "a3z_failed_entries_per_entry",
            "ptv_coalesced_rapm_possession_offense",
        }
    ),
    "cat_shot_defense": frozenset(
        {
            "ptv_coalesced_rapm_threat_suppression",
            "ptv_coalesced_rapm_possession_defense",
            "eh_rapm_CA/60",
            "rapm_hybrid_xGA/60",
            "a3z_carries_with_chance_against_per_target",
            "a3z_carries_with_chance_against_per_60",
        }
    ),
    "cat_transition_defense": frozenset(
        {
            "a3z_denials_per_target",
            "a3z_dz_controlled_breakout_per_60",
            "a3z_retrievals_leading_to_exits_per_60",
            "a3z_passes_allowed_per_target",
            "a3z_failed_exits_per_exit_touch",
            "a3z_dz_retrievals_per_60",
        }
    ),
    "cat_playmaking": frozenset(
        {
            "a3z_expected_assists_proxy_per_60_shrunk",
            "a3z_playmaking_quality_proxy_per_60",
        }
    ),
    "cat_finishing": frozenset(
        {
            "eh_finishing_sh_pct_above_xsh_shrunk",
            "eh_finishing_sh_pct_above_xsh_blended",
            "eh_finishing_goals_above_ixg_per_60_shrunk",
            "eh_finishing_g_per_shot_above_xg_per_shot_shrunk",
        }
    ),
    "cat_forecheck": frozenset(
        {
            "a3z_forecheck_pressures_per_60",
            "a3z_cycle_forecheck_events_per_60",
            "a3z_recoveries_per_60",
        }
    ),
    "cat_physicality": frozenset(
        {
            "eh_std_iHF/60",
            "eh_std_iBLK/60",
            "eh_std_sh_iBLK/60",
        }
    ),
    "cat_skating": frozenset(
        {
            "skating_pillar_speed",
            "skating_pillar_acceleration",
            "skating_pillar_agility",
            "skating_pillar_balance",
            "skating_pillar_endurance",
        }
    ),
    "cat_power_play": frozenset({"eh_std_pp_ixG/60", "eh_std_pp_iCF/60"}),
    "cat_penalty_kill": frozenset({"eh_std_sh_TAKE/60", "eh_std_sh_iPEN±/60"}),
}

COMPONENT_A3Z_MICRO_LINEAGE: dict[str, frozenset[str]] = {
    "a3z_expected_assists_proxy_per_60_shrunk": frozenset(
        {"a3z_chance_assists_per_60", "a3z_primary_shot_assists_per_60"}
    ),
    "a3z_playmaking_quality_proxy_per_60": frozenset(
        {"a3z_home_plate_passes_per_60", "a3z_center_lane_assists_per_60"}
    ),
    "skating_pillar_speed": frozenset(),
    "skating_pillar_acceleration": frozenset(),
    "skating_pillar_agility": frozenset(),
    "skating_pillar_balance": frozenset(),
    "skating_pillar_endurance": frozenset(),
}

COMPONENT_FORBIDDEN_FEATURES: dict[str, frozenset[str]] = {
    "cat_playmaking": frozenset(
        {
            "ptv_coalesced_rapm_pre_shot_pressure",
            "ptv_coalesced_rapm_shot_threat",
            "rapm_hybrid_xGF/60",
            "eh_std_G/60",
            "eh_std_A1/60",
            "eh_std_A2/60",
            "eh_std_Points/60",
        }
    ),
    "cat_finishing": frozenset(
        {
            "eh_std_G/60",
            "eh_std_iCF/60",
            "eh_std_ixG/60",
            "a3z_chances_per_60",
            "a3z_shots_per_60",
        }
    ),
    "cat_physicality": frozenset(
        {
            "a3z_forecheck_pressures_per_60",
            "a3z_cycle_forecheck_events_per_60",
            "a3z_gs_forechecking",
        }
    ),
}


# Raw box-score point/assist counting stats must not enter category components.
# Process metrics (ixG, iCF, RAPM, A3Z micro, G-vs-ixG conversion) are allowed.
FORBIDDEN_BOX_SCORE_POINT_FEATURES: frozenset[str] = frozenset(
    {
        "eh_std_Points/60",
        "eh_std_A1/60",
        "eh_std_A2/60",
        "eh_std_G/60",
        "eh_std_pp_Points/60",
        "eh_std_pp_A1/60",
        "eh_std_pp_A2/60",
    }
)


def assert_no_box_score_points_in_components(
    features: list[str] | set[str],
    *,
    label: str = "category specs",
) -> None:
    """Raise if any forbidden box-score point/assist feature is in a component spec."""
    bad = sorted(set(features) & FORBIDDEN_BOX_SCORE_POINT_FEATURES)
    if bad:
        raise ValueError(
            f"Box-score point/assist features are not allowed in {label}: {', '.join(bad)}"
        )


def assert_isolated_feature_ownership(
    category_features: dict[str, list[str]],
    *,
    label: str = "category specs",
) -> None:
    """Raise if any raw feature appears in more than one category."""
    owners: dict[str, list[str]] = {}
    for cat, feats in category_features.items():
        for feat in feats:
            owners.setdefault(feat, []).append(cat)
    dupes = {f: cs for f, cs in owners.items() if len(cs) > 1}
    if dupes:
        lines = [f"  {feat}: {cats}" for feat, cats in sorted(dupes.items())]
        raise ValueError(
            f"Feature overlap detected in {label} ({len(dupes)} duplicates):\n" + "\n".join(lines)
        )


def assert_component_feature_contract(
    category_features: dict[str, list[str]],
    *,
    label: str = "category specs",
) -> None:
    """Raise if a category uses a feature outside its explicit component contract."""
    assert_no_deprecated_a3z_features(
        (feature for features in category_features.values() for feature in features),
        label=label,
    )
    problems: list[str] = []
    for cat, feats in sorted(category_features.items()):
        allowed = COMPONENT_FEATURE_CONTRACTS.get(cat)
        if allowed is None:
            continue
        unknown = sorted(set(feats) - allowed)
        forbidden = sorted(set(feats) & COMPONENT_FORBIDDEN_FEATURES.get(cat, frozenset()))
        if unknown:
            problems.append(f"  {cat}: uncontracted features: {', '.join(unknown)}")
        if forbidden:
            problems.append(f"  {cat}: forbidden features: {', '.join(forbidden)}")
    if problems:
        raise ValueError(
            f"Component feature contract failed for {label}:\n" + "\n".join(problems)
        )


def _a3z_micro_lineage(feature: str) -> frozenset[str]:
    if feature in COMPONENT_A3Z_MICRO_LINEAGE:
        return COMPONENT_A3Z_MICRO_LINEAGE[feature]
    if feature.startswith("a3z_"):
        return frozenset({feature})
    return frozenset()


def assert_disjoint_a3z_microstats(
    category_features: dict[str, list[str]],
    *,
    label: str = "category specs",
) -> None:
    """Raise if an A3Z source microstat feeds more than one component."""
    owners: dict[str, set[str]] = {}
    for cat, feats in category_features.items():
        for feat in feats:
            for micro in _a3z_micro_lineage(feat):
                owners.setdefault(micro, set()).add(cat)
    dupes = {micro: sorted(cats) for micro, cats in owners.items() if len(cats) > 1}
    if dupes:
        lines = [f"  {micro}: {cats}" for micro, cats in sorted(dupes.items())]
        raise ValueError(
            f"A3Z microstat lineage overlap detected in {label} ({len(dupes)} duplicates):\n"
            + "\n".join(lines)
        )

FORECHECK_PHYSICALITY_ROLE_SPECS: dict[str, dict[str, list[tuple[str, float]]]] = {
    role: {
        "cat_forecheck": specs["Forecheck"],
        "cat_physicality": specs["Physicality"],
    }
    for role, specs in EXTRA_CATEGORY_SPECS_BY_ROLE.items()
}

CAT_COLS = [
    "cat_transition_offense_z",
    "cat_shot_defense_z",
    "cat_transition_defense_z",
    "cat_shot_creation_z",
    "cat_finishing_z",
    "cat_playmaking_z",
    "cat_power_play_z",
    "cat_penalty_kill_z",
    "cat_forecheck_z",
    "cat_physicality_z",
    "cat_skating_z",
]

NOISY_CAT_COLS = {"cat_power_play_z", "cat_penalty_kill_z"}

# Legacy combined ST column (rebuilt from PP/PK for older scripts).
LEGACY_SPECIAL_TEAMS_Z = "cat_special_teams_z"

CAT_ADJ_COLS = [f"{c}__adj" for c in CAT_COLS]

CAT_CTX_COLS = [f"{c}__adj__ctx" for c in CAT_COLS]
