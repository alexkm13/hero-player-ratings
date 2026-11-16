"""Fail-closed integrity rules for A3Z aggregate features.

The source exports have changed definitions across provider eras.  Keep these
checks close to ingestion so an impossible rate cannot be clipped, standardized,
or silently interpreted as player skill downstream.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# ``retrievals_to_exits`` and ``dz_retrieval`` are not a compatible
# numerator/denominator pair in the current aggregate export.  In the modern
# qualified sample, almost every populated quotient exceeds one.  The per-60
# retrievals-to-exits count remains usable; only this derived quotient is banned.
DEPRECATED_A3Z_FEATURES: frozenset[str] = frozenset(
    {"a3z_defensive_exits_after_retrieval_rate"}
)

BOUNDED_A3Z_RATE_STEMS: frozenset[str] = frozenset(
    {
        "denials_per_target",
        "passes_allowed_per_target",
        "carries_with_chance_against_per_target",
        "failed_entries_per_entry",
        "failed_exits_per_exit_touch",
    }
)


def sanitize_a3z_rates(
    frame: pd.DataFrame,
    *,
    prefix: str = "",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a copy with invalid A3Z rates set missing and an audit count.

    This function never clips an impossible value into the legal interval.
    Clipping would turn a source-definition error into apparently extreme skill.
    """

    out = frame.copy()
    audit: dict[str, int] = {}

    deprecated = f"{prefix}defensive_exits_after_retrieval_rate"
    if deprecated in out.columns:
        n_present = int(pd.to_numeric(out[deprecated], errors="coerce").notna().sum())
        audit[f"deprecated:{deprecated}"] = n_present
        out[deprecated] = np.nan

    for stem in sorted(BOUNDED_A3Z_RATE_STEMS):
        col = f"{prefix}{stem}"
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        bad = values.notna() & ~values.between(0.0, 1.0, inclusive="both")
        audit[f"out_of_bounds:{col}"] = int(bad.sum())
        out[col] = values.mask(bad)

    return out, audit


def assert_no_deprecated_a3z_features(
    features: Iterable[str],
    *,
    label: str,
) -> None:
    """Reject model specifications that reintroduce a banned A3Z feature."""

    bad = sorted(set(features) & DEPRECATED_A3Z_FEATURES)
    if bad:
        raise ValueError(
            f"Deprecated A3Z feature(s) in {label}: {', '.join(bad)}. "
            "The source numerator and denominator are not definition-compatible."
        )


def assess_a3z_season_coverage(
    frame: pd.DataFrame,
    *,
    season_col: str = "season",
    toi_col: str = "toi_5v5_min",
    comparable_floor: float = 0.80,
) -> pd.DataFrame:
    """Summarize source exposure and flag low-coverage seasons.

    Coverage is measured against the median of the two immediately preceding
    *observed* seasons.  This is a completeness warning, not an imputation and
    not a claim that tracked games are a random sample.
    """

    if season_col not in frame.columns or toi_col not in frame.columns:
        raise KeyError(f"Coverage audit requires {season_col!r} and {toi_col!r}")

    d = frame[[season_col, toi_col]].copy()
    d[toi_col] = pd.to_numeric(d[toi_col], errors="coerce")
    coverage = (
        d.groupby(season_col, dropna=False, as_index=False)
        .agg(
            source_rows=(season_col, "size"),
            source_toi_5v5_min=(toi_col, "sum"),
            source_rows_with_toi=(toi_col, lambda value: int(value.notna().sum())),
        )
        .sort_values(season_col)
        .reset_index(drop=True)
    )
    totals = pd.to_numeric(coverage["source_toi_5v5_min"], errors="coerce")
    reference = totals.shift(1).rolling(2, min_periods=2).median()
    ratio = totals / reference.replace(0.0, np.nan)
    coverage["source_prior_2yr_median_toi"] = reference
    coverage["source_coverage_ratio"] = ratio
    coverage["source_coverage_status"] = np.where(
        ratio.isna(),
        "UNASSESSED",
        np.where(ratio >= float(comparable_floor), "COMPARABLE", "PARTIAL_UNCERTIFIED"),
    )
    return coverage
