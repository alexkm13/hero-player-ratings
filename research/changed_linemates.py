"""Outcome-blind lineup cohorts; replay existing historical predictions only."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from research.identification.common import prepare_ledger, team
from research.paths import ARCHIVE as ROOT, output

OUT = output('changed_linemates')
AUDIT = ROOT / 'outputs/player_identification_audit_v1'
ACTOR = ROOT / 'outputs/a3z_actor_value_v1'
FOLDS = ((20232024, 20242025), (20242025, 20252026))
KEY = ['game', 'side', 'player_id']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source_profiles(source, pg, players, meta):
    g = np.load(AUDIT / f'geometry_{source}.npz')['same'].copy()
    hours = np.diag(g).copy()
    np.fill_diagonal(g, 0.)
    assert np.allclose(g.sum(axis=1), 4 * hours, atol=1e-7)
    dominant = g.argmax(axis=1)
    shared = g[np.arange(len(players)), dominant]
    p = pd.DataFrame({'player_id': players, 'source_end': source,
        'source_minutes': 60 * hours, 'dominant_mate': players[dominant],
        'shared_minutes': 60 * shared,
        'source_mate_share': np.divide(shared, hours, out=np.zeros_like(hours), where=hours > 0)})
    teams = pg[pg.season.eq(source)].groupby('player_id').agg(
        source_franchises=('franchise', 'nunique'), source_team=('franchise', 'first')).reset_index()
    roles = meta[meta.season_id.eq(source)][['player_id', 'position_group']].drop_duplicates()
    assert not roles.duplicated('player_id').any()
    p = p.merge(teams, on='player_id', how='left', validate='one_to_one').merge(
        roles, on='player_id', how='left', validate='one_to_one')
    p['source_eligible'] = (p.source_minutes.ge(300) & p.shared_minutes.ge(120)
        & p.source_mate_share.ge(.35) & p.position_group.isin(['F', 'D']))
    return p


def target_exposure(target, profile, pg, tg, players):
    q = prepare_ledger(target)
    assert q.season.eq(target).all() and q.seconds.gt(0).all()
    lookup = pd.Index(players)
    dom = profile.set_index('player_id').dominant_mate.reindex(players).to_numpy()
    pieces = []
    for side, col in [(0, 'home_players'), (1, 'away_players')]:
        ids = np.stack(q[col].to_numpy()).astype(np.int64)
        indices = lookup.get_indexer(ids.ravel()).reshape(-1, 5)
        assert (indices >= 0).all()
        together = (dom[indices][:, :, None] == ids[:, None, :]).any(axis=2)
        sec = np.repeat(q.seconds.to_numpy(float), 5)
        known = np.repeat(~q.context_missing.to_numpy(), 5)
        raw = pd.DataFrame({'game': np.repeat(q.game.to_numpy(), 5),
            'player_id': ids.ravel(), 'seconds_check': sec,
            'mate_seconds': sec * together.ravel(),
            'score_seconds': sec * np.repeat(q.score_state.to_numpy(float), 5)
                * (1 if side == 0 else -1) * known,
            'known_seconds': sec * known})
        d = raw.groupby(['game', 'player_id'], sort=False, as_index=False).sum()
        d['side'] = side
        pieces.append(d)
    d = pd.concat(pieces, ignore_index=True)
    d = d.merge(pg[pg.season.eq(target)], on=KEY, validate='one_to_one')
    assert np.allclose(d.seconds, d.seconds_check, atol=1e-8)
    d = d.drop(columns='seconds_check').merge(profile, on='player_id', validate='many_to_one')
    d['target_mate_share'] = d.mate_seconds / d.seconds
    d['mate_share_drop'] = d.source_mate_share - d.target_mate_share
    d['score_mean'] = d.score_seconds / d.known_seconds.replace(0, np.nan)
    d['missing_fraction'] = 1 - d.known_seconds / d.seconds
    assert d.target_mate_share.between(-1e-9, 1 + 1e-9).all()
    assert d.missing_fraction.between(-1e-9, 1 + 1e-9).all()
    d['within_team'] = d.source_franchises.eq(1) & d.franchise.eq(d.source_team)
    target_teams = d.groupby('player_id').franchise.nunique()
    d['clean_mover'] = (d.source_franchises.eq(1)
        & d.player_id.map(target_teams).eq(1) & d.franchise.ne(d.source_team))
    d['eligible'] = d.source_eligible & d.seconds.ge(480)
    d['cohort'] = 'ineligible'
    d.loc[d.eligible, 'cohort'] = 'other_team_history'
    d.loc[d.eligible & d.clean_mover, 'cohort'] = 'clean_mover'
    stay = d.eligible & d.within_team
    d.loc[stay, 'cohort'] = 'intermediate'
    d.loc[stay & d.mate_share_drop.ge(.25), 'cohort'] = 'changed'
    d.loc[stay & d.mate_share_drop.abs().le(.10), 'cohort'] = 'stable'
    opponents = tg[tg.season.eq(target)][['game', 'side', 'franchise']].copy()
    opponents['side'] = 1 - opponents.side
    d = d.merge(opponents.rename(columns={'franchise': 'opponent'}), on=['game', 'side'], validate='many_to_one')
    source = tg[tg.season.eq(target - 10001)].copy()
    against = source[['game', 'side', 'shots']].copy()
    against['side'] = 1 - against.side
    source = source.merge(against.rename(columns={'shots': 'shots_against'}), on=['game', 'side'], validate='one_to_one')
    rates = source.groupby('franchise')[['shots', 'shots_against', 'seconds']].sum()
    rate = (rates.shots - rates.shots_against) * 3600 / rates.seconds
    scale = float(rate.std(ddof=0))
    assert scale > 0
    d['opponent_source_z'] = d.opponent.map((rate - rate.mean()) / scale)
    d['home'] = d.side.eq(0)
    d['minutes'] = d.seconds / 60
    d['season_id'] = target
    return d


def match_games(d):
    """Consumes exposure/context columns exclusively; no outcomes or predictions."""
    pairs = []
    good = d[d.cohort.isin(['changed', 'stable']) & d.missing_fraction.eq(0)
        & d.opponent_source_z.notna() & d.score_mean.notna()]
    for (season, player, home), group in good.groupby(['season_id', 'player_id', 'home'], sort=True):
        stable = group[group.cohort.eq('stable')].sort_values('game').copy()
        for row in group[group.cohort.eq('changed')].sort_values('game').itertuples():
            if stable.empty:
                break
            dz = stable.opponent_source_z - row.opponent_source_z
            ds = stable.score_mean - row.score_mean
            ratio = stable.minutes / row.minutes
            keep = dz.abs().le(.5) & ds.abs().le(.5) & ratio.between(2 / 3, 1.5)
            if not keep.any():
                continue
            dist = (dz / .5) ** 2 + (ds / .5) ** 2 + (np.log(ratio) / np.log(1.5)) ** 2
            ix = dist[keep].idxmin()
            other = stable.loc[ix]
            pairs.append({'season_id': int(season), 'player_id': int(player), 'home': bool(home),
                'changed_game': int(row.game), 'stable_game': int(other.game),
                'opponent_z_difference': float(other.opponent_source_z - row.opponent_source_z),
                'score_difference': float(other.score_mean - row.score_mean),
                'minutes_ratio': float(other.minutes / row.minutes)})
            stable = stable.drop(index=ix)
    return pd.DataFrame(pairs, columns=['season_id', 'player_id', 'home', 'changed_game',
        'stable_game', 'opponent_z_difference', 'score_difference', 'minutes_ratio'])


def cluster_ci(delta, labels, seed):
    x = pd.DataFrame({'delta': np.asarray(delta), 'cluster': np.asarray(labels)})
    a = x.groupby('cluster').delta.agg(['sum', 'count'])
    if len(a) < 2:
        return {'clusters': len(a), 'mean': float(x.delta.mean()) if len(x) else None, 'ci95': None}
    sums, sizes = a['sum'].to_numpy(), a['count'].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(20):
        indices = rng.integers(len(a), size=(100, len(a)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return {'clusters': len(a), 'mean': float(x.delta.mean()),
        'ci95': np.quantile(draws, [.025, .975]).tolist()}


def deviance(y, mu):
    assert np.isfinite(mu).all() and (mu > 0).all()
    return 2 * (np.where(y > 0, y * np.log(np.maximum(y, 1) / mu), 0) - y + mu)


def behavior_scores(d, seed):
    if not len(d):
        return {'rows': 0, 'players': 0}
    y = d['count'].to_numpy(float)
    b = deviance(y, d.baseline_expected.to_numpy(float))
    c = deviance(y, d.actor_expected.to_numpy(float))
    return {'rows': len(d), 'players': int(d.player_id.nunique()), 'games': int(d.game.nunique()),
        'zero_rows': int(d['count'].eq(0).sum()), 'actions': int(y.sum()),
        'baseline_deviance': float(b.mean()), 'actor_deviance': float(c.mean()),
        'delta': float((c - b).mean()), 'improvement_pct': float(100 * (b.mean() - c.mean()) / b.mean()),
        'baseline_prediction_ratio': float(d.baseline_expected.sum() / y.sum()) if y.sum() else None,
        'actor_prediction_ratio': float(d.actor_expected.sum() / y.sum()) if y.sum() else None,
        'game_cluster': cluster_ci(c - b, d.game, seed),
        'player_cluster': cluster_ci(c - b, d.player_id, seed + 1)}


def process_scores(d, b, c, seed):
    if not len(d):
        return {'team_sides': 0}
    eb = (d[b].to_numpy() - d.actual.to_numpy()) ** 2
    ec = (d[c].to_numpy() - d.actual.to_numpy()) ** 2
    return {'team_sides': len(d), 'games': int(d.game.nunique()),
        'baseline_mse': float(eb.mean()), 'candidate_mse': float(ec.mean()),
        'delta': float((ec - eb).mean()), 'game_cluster': cluster_ci(ec - eb, d.game, seed)}


def main():
    if (OUT / 'evaluation.json').exists():
        raise RuntimeError('Completed experiment exists; do not overwrite results.')
    paths = [Path(__file__), ACTOR / 'incidence_coice_predictions.parquet',
        ACTOR / 'incidence_coice_effects.parquet', ACTOR / 'incidence_coice_evaluation.json',
        ACTOR / 'action_incidence.parquet', ACTOR / 'coice_exposure.parquet',
        ROOT / 'outputs/changed_lineup_v1/player_games.parquet',
        ROOT / 'outputs/changed_lineup_v1/team_games.parquet',
        ROOT / 'outputs/challenger_hero_conditioned_v1/heldout_team_games.parquet',
        ROOT / 'work/canonical_v1/player_meta.parquet',
        AUDIT / 'team_game_predictions.parquet']
    paths += [AUDIT / f'geometry_{s}.npz' for s, _ in FOLDS]
    prior_eval = json.loads((ACTOR / 'incidence_coice_evaluation.json').read_text())
    # Parent manifests may contain paths from another machine. The local release
    # snapshots the actual inputs at run start and verifies them after scoring.
    assert prior_eval["folds"]
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': {str(p): sha(p) for p in paths},
        'environment': {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'fitted_new_model': False, 'actual_2026_27_used': False, 'seed': 20261102}
    dump('run_manifest.json', manifest)
    print('Local source inputs hashed.', flush=True)
    pg = pd.read_parquet(ROOT / 'outputs/changed_lineup_v1/player_games.parquet')
    tg = pd.read_parquet(ROOT / 'outputs/changed_lineup_v1/team_games.parquet')
    pg['franchise'] = pg.team.map(team)
    tg['franchise'] = tg.team.map(team)
    players = np.load(AUDIT / 'players.npy')
    meta = pd.read_parquet(ROOT / 'work/canonical_v1/player_meta.parquet')
    profiles, exposures, pairframes = [], [], []
    for source, target in FOLDS:
        p = source_profiles(source, pg, players, meta)
        d = target_exposure(target, p, pg, tg, players)
        pairs = match_games(d)
        profiles.append(p)
        exposures.append(d)
        pairframes.append(pairs)
        print('Exposure and outcome-blind pairs prepared:', target, len(d), len(pairs), flush=True)
    profiles = pd.concat(profiles, ignore_index=True)
    exposure = pd.concat(exposures, ignore_index=True)
    pairs = pd.concat(pairframes, ignore_index=True)
    assert exposure.season_id.max() == 20252026
    assert not exposure.duplicated(KEY).any()
    assert not pairs.duplicated(['season_id', 'player_id', 'changed_game']).any()
    assert not pairs.duplicated(['season_id', 'player_id', 'stable_game']).any()
    assert pairs.opponent_z_difference.abs().le(.5).all()
    assert pairs.score_difference.abs().le(.5).all()
    assert pairs.minutes_ratio.between(2 / 3, 1.5).all()
    # A second matching pass after arbitrary outcomes/predictions are added must be identical.
    mutated = exposure.assign(count=-999, baseline_expected=1e12, actor_expected=-1e12)
    pd.testing.assert_frame_equal(pairs, match_games(mutated))
    pred = pd.read_parquet(ACTOR / 'incidence_coice_predictions.parquet')
    panel = pd.read_parquet(ACTOR / 'action_incidence.parquet')
    assert not pred.duplicated(KEY + ['action_type']).any()
    truth = panel[panel.season_id.isin([t for _, t in FOLDS])]
    replay = pred.merge(truth[KEY + ['action_type', 'count', 'hours']], on=KEY + ['action_type'],
        validate='one_to_one', suffixes=('', '_original'))
    assert len(replay) == len(pred) == len(truth)
    assert replay['count'].eq(replay.count_original).all()
    assert np.allclose(replay.hours, replay.hours_original, atol=1e-12)
    effects = pd.read_parquet(ACTOR / 'incidence_coice_effects.parquet')
    effects['season_id'] = effects.source_end + 10001
    keys = effects[['season_id', 'player_id', 'action_type']].assign(source_actor=True)
    d = pred.merge(exposure.drop(columns=['team', 'hours'], errors='ignore'),
        on=['season_id'] + KEY, validate='many_to_one')
    assert len(d) == len(pred) and np.allclose(d.hours * 3600, d.seconds, atol=1e-8)
    d = d.merge(keys, on=['season_id', 'player_id', 'action_type'], how='left', validate='many_to_one')
    assert len(d) == len(pred)
    d['source_actor'] = d.source_actor.eq(True)
    for label, gamecol in [('matched_changed', 'changed_game'), ('matched_stable', 'stable_game')]:
        k = pairs[['season_id', 'player_id', gamecol]].rename(columns={gamecol: 'game'})
        d = d.merge(k.assign(**{label: True}), on=['season_id', 'player_id', 'game'], how='left', validate='many_to_one')
        d[label] = d[label].eq(True)
    reports = []
    for source, target in FOLDS:
        f = d[d.season_id.eq(target) & d.eligible & d.source_actor]
        subsets = {'all_source_eligible': f, 'changed': f[f.cohort.eq('changed')],
            'stable': f[f.cohort.eq('stable')], 'intermediate': f[f.cohort.eq('intermediate')],
            'clean_mover': f[f.cohort.eq('clean_mover')],
            'matched_changed': f[f.matched_changed], 'matched_stable': f[f.matched_stable]}
        for name in ['changed', 'matched_changed', 'clean_mover']:
            for action in ['entry', 'retrieval']:
                subsets[name + '_' + action] = subsets[name][subsets[name].action_type.eq(action)]
        scores = {name: behavior_scores(x, 20261102 + target + j * 2)
            for j, (name, x) in enumerate(subsets.items())}
        retained = pairs[pairs.season_id.eq(target)]
        matched_actor = subsets['matched_changed']
        scores['matched_support'] = {'exposure_pairs_all_games': len(retained),
            'players_all_games': int(retained.player_id.nunique()),
            'pairs_with_tracked_changed_game': int(matched_actor[KEY].drop_duplicates().shape[0]),
            'matched_changed_players': int(matched_actor.player_id.nunique()),
            'max_abs_opponent_z_difference': float(retained.opponent_z_difference.abs().max()) if len(retained) else None,
            'max_abs_score_difference': float(retained.score_difference.abs().max()) if len(retained) else None,
            'mean_abs_opponent_z_difference': float(retained.opponent_z_difference.abs().mean()) if len(retained) else None,
            'mean_abs_score_difference': float(retained.score_difference.abs().mean()) if len(retained) else None}
        reports.append({'source_end': source, 'target': target, 'slices': scores})
    # Behavior pairs need tracking on BOTH games for the paired matched sensitivity.
    # Unpaired matched membership is retained as diagnostic; exact paired re-score below.
    both = pairs.merge(d[KEY + ['season_id', 'action_type', 'count', 'baseline_expected', 'actor_expected',
        'eligible', 'source_actor']], left_on=['season_id', 'player_id', 'changed_game'],
        right_on=['season_id', 'player_id', 'game'], validate='one_to_many')
    both = both.rename(columns={c: c + '_changed' for c in ['game', 'count', 'baseline_expected', 'actor_expected', 'eligible', 'source_actor']})
    both = both.merge(d[KEY + ['season_id', 'action_type', 'count', 'baseline_expected', 'actor_expected',
        'eligible', 'source_actor']], left_on=['season_id', 'player_id', 'stable_game', 'action_type'],
        right_on=['season_id', 'player_id', 'game', 'action_type'], validate='one_to_one', suffixes=('_changed', '_stable'))
    both = both[both.eligible_changed & both.source_actor_changed & both.eligible & both.source_actor]
    for report in reports:
        p = both[both.season_id.eq(report['target'])]
        scored = pd.DataFrame({'game': p.changed_game, 'player_id': p.player_id,
            'count': p.count_changed, 'baseline_expected': p.baseline_expected_changed,
            'actor_expected': p.actor_expected_changed})
        report['slices']['matched_both_tracked_changed'] = behavior_scores(scored, report['target'] + 20261102)
        if len(p):
            dc = deviance(p.count_changed.to_numpy(), p.actor_expected_changed.to_numpy()) - deviance(p.count_changed.to_numpy(), p.baseline_expected_changed.to_numpy())
            ds = deviance(p['count'].to_numpy(), p.actor_expected.to_numpy()) - deviance(p['count'].to_numpy(), p.baseline_expected.to_numpy())
            report['slices']['matched_advantage_change'] = {'paired_action_rows': len(p),
                'changed_minus_stable_delta': float((dc - ds).mean()),
                'player_cluster': cluster_ci(dc - ds, p.player_id, report['target'] + 99)}
    # Team fractions use all five player slots, not just qualifying players or tracked games.
    e = exposure.copy()
    e['supported_seconds'] = e.seconds * (e.source_eligible & e.within_team)
    e['changed_seconds'] = e.seconds * e.cohort.eq('changed')
    team_exposure = e.groupby(['season_id', 'game', 'side'], as_index=False)[['seconds', 'supported_seconds', 'changed_seconds']].sum()
    team_exposure['source_support'] = team_exposure.supported_seconds / team_exposure.seconds
    team_exposure['changed_share'] = team_exposure.changed_seconds / team_exposure.seconds
    team_exposure['process_cohort'] = 'intermediate'
    team_exposure.loc[team_exposure.changed_share.ge(.25), 'process_cohort'] = 'high_change'
    team_exposure.loc[team_exposure.changed_share.le(.05), 'process_cohort'] = 'low_change'
    team_exposure.loc[team_exposure.source_support.lt(.60), 'process_cohort'] = 'low_support'
    bridge = pd.read_parquet(ROOT / 'outputs/challenger_hero_conditioned_v1/heldout_team_games.parquet')
    bridge = bridge.merge(team_exposure, left_on=['season', 'game', 'side'], right_on=['season_id', 'game', 'side'], validate='one_to_one', suffixes=('', '_skaters'))
    assert np.allclose(bridge.seconds_skaters, 5 * bridge.seconds)
    process = {}
    for _, target in FOLDS:
        f = bridge[bridge.season.eq(target)]
        process[str(target)] = {name: process_scores(x, 'hero_categories', 'hero_categories_shares', target + 7)
            for name, x in [('all', f)] + [(c, f[f.process_cohort.eq(c)]) for c in ['high_change', 'low_change', 'low_support']]}
    ridge = pd.read_parquet(AUDIT / 'team_game_predictions.parquet')
    ridge = ridge[ridge.phase.eq('historical_source')].merge(team_exposure, left_on=['season', 'game', 'side'], right_on=['season_id', 'game', 'side'], validate='one_to_one')
    sensitivity = {name: process_scores(x, 'p20_s20', 'p20_s80', 20261104)
        for name, x in [('all', ridge)] + [(c, ridge[ridge.process_cohort.eq(c)]) for c in ['high_change', 'low_change', 'low_support']]}
    support = True
    for r in reports:
        primary = r['slices']['changed']
        matched = r['slices']['matched_both_tracked_changed']
        support &= (primary.get('delta', 1) < 0 and matched.get('delta', 1) < 0 and matched['players'] >= 50
            and all(primary[k]['ci95'] is not None and primary[k]['ci95'][1] < 0 for k in ['game_cluster', 'player_cluster']))
    result = {'status': 'historical_development', 'folds': reports, 'team_process': process,
        'ridge_system_penalty_sensitivity_2025_26': sensitivity,
        'equal_fold_changed_delta': float(np.mean([r['slices']['changed'].get('delta', np.nan) for r in reports])),
        'behavior_support_rule_pass': bool(support), 'new_rating_or_value_claim': False}
    profiles.to_parquet(OUT / 'source_profiles.parquet', index=False)
    exposure.to_parquet(OUT / 'target_exposure.parquet', index=False)
    pairs.to_parquet(OUT / 'matched_pairs.parquet', index=False)
    both.to_parquet(OUT / 'matched_behavior_pairs.parquet', index=False)
    d.to_parquet(OUT / 'behavior_predictions.parquet', index=False)
    bridge.to_parquet(OUT / 'process_predictions.parquet', index=False)
    ridge.to_parquet(OUT / 'ridge_sensitivity_predictions.parquet', index=False)
    for p, h in manifest['files'].items():
        assert sha(p) == h, p
    dump('verification.json', {'status': 'PASS', 'release_runs_separately_from_production': True,
        'source_and_script_hashes_unchanged': True, 'outcome_prediction_matching_invariance': True,
        'full_risk_exposure_reconciliation': True, 'five_slot_team_denominator': True,
        'match_calipers_and_no_reuse': True, 'zero_action_rows_preserved': int(d['count'].eq(0).sum()),
        'target_player_games': len(exposure), 'saved_prediction_rows': len(d),
        'actual_2026_27_used': False, 'production_changed': False})
    dump('evaluation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
