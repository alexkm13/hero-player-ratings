# HERO player ratings

HERO is an NHL skater rating project. The question behind it is pretty simple: how good is a player right now, and how much of what we see travels when his linemates or team change?

The main rating pools three completed seasons of category scores. Recent minutes count more, and short histories get pulled toward the middle. The categories cover things like shot creation, transition play, defense, finishing and passing. The next-season projection is a separate model. The experiments here are ways of checking and improving those pieces; they don't feed into the published rating automatically.

## What's here

- `hero/overall.py` builds the three-season HERO rating from tuned player-season scores and the frozen role weights.
- `pcv/` is the possession-value and adjusted plus-minus pipeline. It has play-by-play and shift ingestion, expected threat, feature and category builders, and projection code.
- `hero/live/` is the first-20-games experiment. It rebuilds 5v5 exposures from shifts, estimates context-adjusted shot differential and tests how much that early evidence should move a preseason prior.
- `research/` has the structural and staged challengers, the lineup identification audit, and the tests that separate entry volume, who takes the entry and whether it crosses.

## Running it

Use Python 3.11 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m hero.overall --help
python pcv/build_rapm_pcv.py --help
```

To rebuild Overall with your own inputs:

```sh
python -m hero.overall \
  --categories data/hero_v8/tuned_categories.parquet \
  --weights data/hero_v8/weights_v7.json \
  --card-list data/hero_v8/current_ratings.csv \
  --cutoff 20252026 \
  --output artifacts/overall.csv
```

The category input needs one row per player and season, with NHL ID, name, position group, eligibility, ice time and the nine category scores listed in `hero/overall.py`. The weights file has a `weights_F` and `weights_D` map keyed by those same columns. The optional card-list argument fixes the percentile cohort by player ID. Leave it off to rank everyone eligible at the cutoff.

The live experiment uses NHL standings and public [fastRhockey play-by-play](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nhl_pbp_lite) and [shift](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nhl_shifts) downloads. These commands reproduce the November 15, 2026 cutoff, subject to revisions in the public source:

```sh
python -m hero.live.bulk
python -m hero.live.fetch
python -m hero.live.prepare
python -m hero.live.evaluate
```

The last step also needs the three local HERO files shown above. It writes an experimental shot estimate alongside the existing rating, in separate columns. For the saved 20-game study, the weight rose with each player's 5v5 minutes and was capped at 30%. That figure applies to the shot-process update; it isn't a weight for every HERO category.

The deeper experiments use local upstream tables. Set `HERO_CANONICAL_ROOT` to the canonical shift/event tables, `HERO_PCV_ROOT` to the PCV tree with your inputs, and `HERO_PRIOR_ROOT` to the earlier run outputs used by the structural and staged models. The archive-backed replays (including `python -m research.entry` and `python -m research.changed_linemates`) take `HERO_ARCHIVE_ROOT`, a local archive with the original `outputs/` and `work/` layout. Their input paths are listed at the top of each module and in its runner. `HERO_DATA_ROOT` and `HERO_ARTIFACTS_ROOT` override the default `data/` and `artifacts/` directories.

This is a source-only checkout. Bring your own tracking exports and fitted inputs. Player records, vendor exports, model outputs and card images stay local. There are still real attribution limits here: ridge can account for who was on the ice, but it can't tell us every reason a coach put them there. Taking an entry and creating a better shot from it are different questions.
