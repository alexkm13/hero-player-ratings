# HERO player ratings

HERO is an NHL skater rating project. The question behind it is pretty simple: how good is a player right now, and how much of what we see travels when his linemates or team change?

The main rating pools three completed seasons of category scores. Recent minutes count more, and short histories get pulled toward the middle. The categories cover things like shot creation, transition play, defense, finishing and passing. The next-season projection is a separate model. The experiments here are ways of checking and improving those pieces; they don't feed into the published rating automatically.

## What's here

- `hero/overall.py` builds the three-season HERO rating from tuned player-season scores and the frozen role weights.
- `pcv/` is the possession-value and adjusted plus-minus pipeline. It has play-by-play and shift ingestion, expected threat, feature and category builders, and projection code.
- `hero/live/` is the first-20-games experiment. It rebuilds 5v5 exposures from shifts, estimates context-adjusted shot differential and tests how much that early evidence should move a preseason prior.
- `research/` has the structural and staged challengers, the lineup identification audit, and the tests that separate entry volume, who takes the entry and whether it crosses.

## How HERO compares

There are two different questions here: **does a rating stay stable from year to year, and does it tell us anything about what happens next?** We report both. No test below treats a later rating as ground truth for player talent.

### The public WAR comparison that prompted this

Aaron Knodell's [*Which Public WAR Model Is Most Predictive?*](https://puckovertheglass.substack.com/p/which-public-war-model-is-most-predictive) compared [Evolving-Hockey WAR and xWAR](https://evolving-hockey.com/glossary/goals-above-replacement/), [HockeyStats / TopDownHockey WAR](https://hockeystats.com/methodology/war), and [LB-Hockey SPAR](https://lb-hockey.com/free-tools/). These are **his results, not a test that included HERO**. The player test takes the top 384 forwards and 192 defensemen by ice time per season, keeps players who qualify in both consecutive seasons, and compares each rating to its own next-year value. He uses per-60 scores and reports the correlation separately by position.

| Public rating | Forward Pearson r | Forward r² | Defense Pearson r | Defense r² | Team points r | Team points r² |
|---|---:|---:|---:|---:|---:|---:|
| Evolving-Hockey WAR | .383 | .147 | .297 | .088 | .609 | .371 |
| Evolving-Hockey xWAR | .438 | .192 | .464 | .215 | .635 | .403 |
| HockeyStats WAR | .560 | .313 | .469 | .220 | — | — |
| LB-Hockey SPAR | .725 | .526 | .652 | .425 | .687 | .473 |

In the article's team test (2022–23 through 2025–26), prior-season SPAR rates are applied to actual target-season ice time. The points target subtracts Evolving-Hockey's estimates for goalies and skaters outside the regular-player cohort. HockeyStats was left out because the author didn't have a shared player key for joining it to the other sources. This is not the same outcome or setup as our 5v5 shot test.

### Our test against future 5v5 play: LB-Hockey and Hockey Alchemy

We use **actual published scores** from the LB public SPAR table and the August 2026 [Hockey Alchemy](https://www.hockeyalchemy.com/methodology) GAR snapshot, alongside the frozen HERO V8 rating. Hockey Alchemy and HockeyStats are different providers. The completed-season ratings predict team 5v5 unblocked shots, [MoneyPuck xG](https://moneypuck.com/data.htm), and goals the following season.

All rating inputs get the same forecast structure: home ice, prior-game shot form, rating coverage, and four rating features: own forwards, own defensemen, opposing forwards, and opposing defensemen. We condition on **actual 5v5 deployment**. The main competitor arms divide GAR or SPAR by source all-strength hours; the raw published totals are a sensitivity check. For the main comparison, both teams need at least 80% of their skater-seconds covered by the same rated-player cohort. Source players need at least 300 canonical 5v5 minutes. Position and source-season scaling use the common rated players. Ridge tuning and fitting use only earlier seasons: train on 2022–23 and validate on 2023–24 before the first held-out test, then roll forward one season. Refit the selected adapter on all earlier seasons. The result uses **1,682 games (3,364 team sides)**. The simple baseline has home ice only; the stronger baseline has form and coverage but no player rating.

| Rating input | Unblocked-shot MSE ↓ | xG MSE ↓ | Goal-count MSE ↓ |
|---|---:|---:|---:|
| Home ice only | 54.141638 | .438984 | 1.761007 |
| Form and coverage, no player rating | 49.141899 | .423902 | 1.758162 |
| HERO V8 | **48.171273** | **.418237** | **1.756254** |
| LB SPAR/hour | 48.942161 | .421727 | 1.760363 |
| Hockey Alchemy GAR/hour | 48.964123 | .422390 | 1.759803 |
| LB raw SPAR | 48.684203 | .420822 | 1.757680 |
| Hockey Alchemy raw GAR | 48.827660 | .422049 | 1.761464 |

These are equal-season means of team-side squared error. HERO lowers shot MSE by **1.58% vs. LB**, **1.62% vs. Hockey Alchemy**, and **1.98% vs. the form baseline**. It wins on shots in both held-out seasons. The goal differences are uncertain.

| Target season | Games | Form baseline shot MSE | HERO | LB rate | Hockey Alchemy rate |
|---|---:|---:|---:|---:|---:|
| 2024–25 | 889 | 52.265872 | 51.236910 | 52.338371 | 52.339444 |
| 2025–26 | 793 | 46.017926 | 45.105635 | 45.545951 | 45.588802 |

On the same source rating cohort, we also ask which players had better actual on-ice rates the following year. This diagnostic uses **1,133 player-seasons** with at least 300 target 5v5 minutes, including **142 clean team moves (133 players)**. A clean move means one franchise in each season, with a different franchise in the target year. Values below are Spearman rank correlations, averaging forward/defense and season cells equally.

| Next-year outcome | HERO V8 | LB rate | Hockey Alchemy rate |
|---|---:|---:|---:|
| All: unblocked-shot net/60 | .477 | .443 | .423 |
| All: xG net/60 | .440 | .401 | .390 |
| All: goal net/60 | .358 | .326 | .352 |
| Clean movers: unblocked-shot net/60 | .348 | .262 | .204 |
| Clean movers: xG net/60 | .285 | .216 | .141 |
| Clean movers: goal net/60 | .237 | .123 | .147 |

<details>
<summary>Uncertainty, coverage, and sensitivity checks</summary>

| Rating contrast (HERO minus competitor) | Shot MSE difference, 97.5% paired-game interval | xG MSE difference, 97.5% interval | Goal MSE difference, 97.5% interval |
|---|---|---|---|
| Versus LB rate | −.770888 [−1.195401, −.343996] | −.003490 [−.006689, −.000480] | −.004109 [−.010097, +.001982] |
| Versus Hockey Alchemy rate | −.792851 [−1.254899, −.324980] | −.004153 [−.007675, −.001006] | −.003550 [−.007686, +.000549] |

- The 2,000 bootstrap draws keep both sides of each game together and sample within season. A team-season clustering check also favors HERO on shots and xG. Pooled shot MAE is 5.528 for HERO, 5.580 for LB rate, 5.586 for Hockey Alchemy rate, and 5.594 for form. HERO's actual/predicted shot ratios are .987 and .988 by season.
- For future player shot-net rank, the 95% player-cluster interval for HERO minus LB is [−.0054, +.0741] overall and [−.0345, +.2079] on movers. Against Hockey Alchemy it is [+.0080, +.0982] overall and [−.0442, +.3152] on movers. The mover intervals are too wide to establish better portable individual value.
- On 988 team sides (831 games) with exactly one qualifying mover playing at least eight 5v5 minutes, pooled shot MSE is 47.041 for HERO, 47.748 for LB rate, and 47.625 for Hockey Alchemy rate. The LB contrast favors HERO when clustering by game; it narrowly includes zero when clustering by mover.
- At 90% common coverage (385 games, same fitted models), HERO still has the lowest shot MSE; LB rate has slightly lower xG MSE (.391951 vs. .392682, pooled). On the full cohort, raw LB SPAR has higher future player goal-net correlation than HERO (.367 vs. .358 overall; .249 vs. .237 for movers). The raw-score results matter because dividing a counting metric by ice time changes what it measures.
- Aggregating the same predictions to team-season shot net/60 gives prediction R² of .707/.805 for HERO, .692/.780 for LB rate, .695/.787 for Hockey Alchemy rate and .707/.764 for the form baseline in 2024–25/2025–26. These are conditional shot-rate results, not standings projections.
- The LB skater table contained 4,172 rows, of which 4,114 had accepted identities. We matched normalized names within role and season, checked team against independent lineup data and dropped ambiguous cases. The 80% threshold was fixed before scoring because a 90% threshold left too few games to fit the first season. The 80% rule kept 445/605 games in the two calibration years and 889/793 in the held-out years. It selects toward established rosters.

</details>

### Our matched-player test with Evolving-Hockey

We found local April 12, 2026 exports with actual EH WAR and xWAR, not just RAPM. We match by NHL ID, role, season and complete team-stint coverage, sum eligible stints and divide by EH all-strength ice time. The April export's incomplete 2025–26 is never a source rating in this test. Earlier source seasons can still predict the *completed* 2025–26 NHL outcomes. The EH files don't cover enough common players for a reliable team test: **zero** games in either target season meet the original both-sides 80% rule. We keep the rule and report the player test separately.

This smaller cohort has **469 player-seasons (306 skaters)**: 241 in 2024–25 and 228 in 2025–26. Each row has all five ratings, plus at least 300 canonical 5v5 minutes in both the source and target seasons. The scale is the same Spearman diagnostic as above; the sample is different.

| Next-year on-ice outcome | HERO V8 | LB rate | Hockey Alchemy rate | EH WAR/60 | EH xWAR/60 |
|---|---:|---:|---:|---:|---:|
| Unblocked-shot net/60 | **.496** | .398 | .481 | .379 | .324 |
| xG net/60 | .445 | .367 | .445 | .330 | .307 |
| Goal net/60 | .368 | .341 | **.382** | .279 | .249 |

For shot net/60, 95% paired player-cluster intervals for HERO's correlation difference are **[+.034, +.193]** vs. EH WAR/60, **[+.098, +.243]** vs. EH xWAR/60, [+.037, +.157] vs. LB, and [−.053, +.081] vs. Hockey Alchemy. The EH total-score sensitivity raises their shot correlations to .397 (WAR) and .364 (xWAR). There are 50 clean-mover player-seasons, but the two defense cells have only 9 and 7 rows. We don't quote a combined mover estimate on this smaller cohort.

<details>
<summary>Year-to-year persistence on our common sample</summary>

To connect back to the article, we also applied its regular-player idea using the archived all-strength ice time: select the top 384 forwards and 192 defensemen first, then require coverage for all ratings in two consecutive seasons. We use 2021–22→2022–23, 2022–23→2023–24, and 2023–24→2024–25, with role/season standardization before pairing. This yields **314 forward pairs and 138 defense pairs**. Different source files, rate denominators and missing-player exclusions mean this is not an exact replication of Knodell's sample.

| Rating | Forward r | Forward r² | Defense r | Defense r² |
|---|---:|---:|---:|---:|
| HERO V8 | .924 | .854 | .896 | .803 |
| LB rate | .698 | .488 | .674 | .454 |
| Hockey Alchemy rate | .714 | .510 | .774 | .599 |
| EH WAR/60 | .405 | .164 | .249 | .062 |
| EH xWAR/60 | .451 | .204 | .523 | .274 |
| HockeyStats WAR | — | — | — | — |

HERO reuses two seasons of evidence in consecutive three-season ratings. That overlap makes a high correlation unsurprising. This table measures stability and should not be used to claim HERO is more accurate. For HockeyStats we have the article's published aggregate results, not a matched historical export, so there is no internal HockeyStats score.

</details>

**What we take from this:** HERO adds a modest, repeatable shot signal on the tested lineups. We have much less evidence for goals or for how much of the signal belongs to a player in a new environment. Ratings use different lookback windows and may include special teams. The 5v5 player outcomes still reflect team and deployment effects. Our historical inputs are revised or reconstructed (August 2026 Hockey Alchemy archive, October 2026 LB revision, historical HERO settings not certified as of each cutoff). These are retrospective tests of shared adapters and player ordering. We haven't tested the providers' native forecasts or established what wins prospectively. The separate 2026–27 benchmark was frozen with a November 1 start; no games from it were scored here.

**Reproduce the tables:** [team/player benchmark code](research/public_benchmark.py) and [EH comparison code](research/eh_benchmark.py). With the original local archive and separately acquired vendor exports:

```sh
export HERO_ARCHIVE_ROOT=/path/to/local/research-archive
python -m research.public_benchmark
python -m research.eh_benchmark --war /path/to/eh-gar.csv --xwar /path/to/eh-xgar.csv
```

The replay reads the frozen common `source_scores.parquet` and `player_targets.parquet` in the archive's `outputs/hero_competitor_benchmark_v1/` and the canonical game tables in `outputs/changed_lineup_v1/`. The EH persistence check also reads the archived Hockey Alchemy TOI table. No player-level data, provider exports or generated predictions are included here.

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
