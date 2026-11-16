"""Source-matched evidence proxies for HERO; unknown is distinct from zero."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRACKED_MINUTES = "a3z_tracked_toi_5v5_min"
EXPOSURE_SOURCE_COLUMNS = (
    TRACKED_MINUTES, "a3z_tracking_observed", "has_a3z_season",
    "a3z_chance_assists", "a3z_primary_shot_assists", "a3z_retrievals_to_exits",
    "a3z_chance_assists_per_60", "a3z_primary_shot_assists_per_60",
    "a3z_retrievals_leading_to_exits_per_60", "eh_std_TOI",
    "eh_std_iSF/60", "eh_std_iHF/60", "eh_std_iBLK/60", "eh_std_TAKE/60",
    "eh_std_strength_scope",
)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    value = df[column] if column in df else pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)


def tracked_minutes(df: pd.DataFrame) -> pd.Series:
    """An explicit tracked field is authoritative, including its missing values."""
    column = TRACKED_MINUTES if TRACKED_MINUTES in df else "a3z_toi_5v5_min"
    minutes = numeric(df, column)
    observed = minutes.gt(0)
    for flag in ("has_a3z_season", "a3z_tracking_observed"):
        if flag in df:
            observed &= df[flag].eq(True)
    return minutes.where(observed)


def preserve_tracking(df: pd.DataFrame) -> None:
    """Call before compatibility code substitutes general ice time."""
    minutes = tracked_minutes(df)
    df[TRACKED_MINUTES] = minutes
    df["a3z_tracking_observed"] = minutes.notna()
    df["a3z_toi_is_fallback"] = minutes.isna() & numeric(df, "a3z_toi_5v5_min").gt(0)


def a3z_count(df: pd.DataFrame, count_column: str, rate_column: str) -> pd.Series:
    minutes = tracked_minutes(df)
    count = numeric(df, count_column)
    count = count.where(count.ge(0))
    # Source rates may be adjusted; original counts take precedence when known.
    reconstructed = numeric(df, rate_column) * minutes / 60.0
    reconstructed = reconstructed.where(reconstructed.ge(0))
    return count.where(count.notna(), reconstructed).where(minutes.notna())


def eh_standard_count(df: pd.DataFrame, rate_column: str) -> pd.Series:
    """The standard table's own TOI is the only valid rate denominator here."""
    minutes = numeric(df, "eh_std_TOI")
    rate = numeric(df, rate_column)
    return (rate * minutes / 60.0).where(minutes.gt(0) & rate.ge(0))


def exposure_proxies(df: pd.DataFrame) -> dict[str, pd.Series]:
    ca = a3z_count(df, "a3z_chance_assists", "a3z_chance_assists_per_60")
    psa = a3z_count(df, "a3z_primary_shot_assists", "a3z_primary_shot_assists_per_60")
    retrieval = a3z_count(df, "a3z_retrievals_to_exits", "a3z_retrievals_leading_to_exits_per_60")
    return {
        "finishing_shots": eh_standard_count(df, "eh_std_iSF/60"),
        "playmaking_weighted_assists": .60 * ca + .40 * psa,
        "playmaking_assists": ca + psa,
        "transition_defense_retrievals": retrieval,
        "physicality_events": (eh_standard_count(df, "eh_std_iHF/60") +
                               eh_standard_count(df, "eh_std_iBLK/60") +
                               eh_standard_count(df, "eh_std_TAKE/60")),
    }
