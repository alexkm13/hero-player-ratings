"""Construct one comparable strict 5v5 shot/shift ledger from bulk NHL releases."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hero.live.fetch import RAW, OUT, CUT
from hero.common import team, sha

WORK=OUT / "ledger"
PBP=['game_id','game_date','event_id','event_type','period','period_seconds','event_team_abbr',
     'home_skaters','away_skaters','home_goalie_id','away_goalie_id',
     'event_goalie_id','event_player_1_id','empty_net','home_abbr','away_abbr',
     *[f'{side}_on_{i}_id' for side in ('home','away') for i in range(1,6)]]
INTERVAL_COLUMNS=['game','date','season','home_team','away_team','t0','t1','seconds',
                  'home_players','away_players','score','period','home_shots','away_shots']
SHOTS={'SHOT','MISSED_SHOT','GOAL'}
ENDS=range(2022,2028)


def tokens(s):
    if isinstance(s,str):
        return {int(x.strip()) for x in s.split(',') if x.strip().isdigit() and int(x.strip())>0}
    return set()


def shot_time(q):
    return (int(q.period)-1)*1200+int(q.period_seconds)


def prep_game(g, pbp, shifts, season):
    home,away=team(g.home_team_abbr),team(g.away_team_abbr)
    assert pbp.home_abbr.map(team).eq(home).all() and pbp.away_abbr.map(team).eq(away).all(),g.game_id
    pbp=pbp[pbp.period.between(1,3)]
    shifts=shifts[shifts.period.between(1,3)]
    hcols=[f'home_on_{i}_id' for i in range(1,6)]+['home_goalie_id']
    acols=[f'away_on_{i}_id' for i in range(1,6)]+['away_goalie_id']
    hsv,asv=({int(i) for i in pd.unique(pbp[c].to_numpy().ravel()) if pd.notna(i) and i>0}
             for c in [hcols,acols])
    goalies={int(i) for i in pd.unique(pbp[['home_goalie_id','away_goalie_id','event_goalie_id']].to_numpy().ravel()) if pd.notna(i) and i>0}
    changes=defaultdict(list)
    labeled=[(s,tokens(s.ids_on),tokens(s.ids_off)) for s in shifts.itertuples(index=False)]
    by_label=defaultdict(set)
    for s,add,remove in labeled:
        by_label[s.event_team].update(add|remove)
    label_to_side={}
    for label,ids in by_label.items():
        h=len(ids & (hsv-asv))
        a=len(ids & (asv-hsv))
        if h==a==0:
            continue # No players belong to either schedule/PBP roster.
        assert h!=a and min(h,a)==0,(g.game_id,label,h,a)
        label_to_side[label]=0 if h>a else 1
    assert set(label_to_side.values())=={0,1},g.game_id
    for s,add,remove in labeled:
        if not (add|remove) or s.event_team not in label_to_side:
            continue
        clock=(int(s.period)-1)*1200+int(s.period_seconds)
        if 0<=clock<3600 and 0<=s.period_seconds<1200:
            changes[clock].append((label_to_side[s.event_team],add,remove))
    goalchanges=defaultdict(int)
    for q in pbp[pbp.event_type.eq('GOAL')].itertuples(index=False):
        side=team(q.event_team_abbr)
        assert side in (home,away),(g.game_id,q)
        # A goal at the end of a period changes score for the next period.
        goalchanges[shot_time(q)]+=1 if side==home else -1
    bounds=sorted(set(changes)|set(goalchanges)|{0,1200,2400,3600})
    bounds=[x for x in bounds if 0<=x<=3600]
    active=[set(),set()]
    score=0
    rows=[]
    states={}
    for t0,t1 in zip(bounds,bounds[1:]):
        if t0 in (0,1200,2400):
            active=[set(),set()]
        for side,add,remove in changes.get(t0,[]):
            active[side].difference_update(remove)
            active[side].update(add)
        score+=goalchanges.get(t0,0)
        h,a=[tuple(sorted(x-goalies)) for x in active]
        hg,ag=[x&goalies for x in active]
        if len(h)!=5 or len(a)!=5 or len(hg)!=1 or len(ag)!=1 or (set(h)&set(a)) or hg==ag:
            continue
        i=len(rows)
        states[t0]=i,hg,ag
        rows.append([int(g.game_id),str(g.game_date),season,home,away,t0,t1,t1-t0,h,a,
                     int(np.clip(score,-2,2)),t0//1200+1,0,0])
    date=str(g.game_date)
    loses=Counter()
    raw=pbp[pbp.event_type.isin(SHOTS)&pbp.home_skaters.eq(5)&pbp.away_skaters.eq(5)&
            pbp.home_goalie_id.notna()&pbp.away_goalie_id.notna()&~pbp.empty_net.fillna(False).astype(bool)]
    st=np.array(bounds)
    for q in raw.itertuples(index=False):
        t=shot_time(q)
        atk=team(q.event_team_abbr)
        if atk not in [home,away] or pd.isna(q.event_player_1_id):
            loses['missing_actor_or_team']+=1
            continue
        side=int(atk==away)
        pbph=tuple(sorted(int(x) for x in [getattr(q,v) for v in hcols[:-1]] if pd.notna(x) and x>0))
        pbpa=tuple(sorted(int(x) for x in [getattr(q,v) for v in acols[:-1]] if pd.notna(x) and x>0))
        tix=st.searchsorted(t,side='right')-1
        opts=[st[tix]] if 0<=tix<len(st) else []
        if tix>0 and st[tix]==t and t!=0:
            opts.insert(0,st[tix-1]) # Prefer pre-play shifts at tied clocks unless the event roster favors the right.
        candidates=[]
        for u in opts:
            if u not in states:
                continue
            i,hg,ag=states[u]
            r=rows[i]
            if r[11]!=q.period:
                continue
            if int(q.event_player_1_id) not in r[8+side]:
                continue
            if pd.notna(q.event_goalie_id) and int(q.event_goalie_id) not in (ag if side==0 else hg):
                continue
            if int(q.home_goalie_id) not in hg or int(q.away_goalie_id) not in ag:
                continue
            candidates.append((int((r[8],r[9])!=(pbph,pbpa)),i))
        if not candidates:
            loses['no_consistent_boundary']+=1
            continue
        # Python min is stable: same roster-match tier uses the earlier interval.
        rank,id=min(candidates,key=lambda x:x[0])
        if rank:
            loses['counted_roster_disagrees']+=1
        rows[id][12+side]+=1
    minutes=sum(row[7] for row in rows)/60
    raw_n=len(raw)
    count=sum(row[12]+row[13] for row in rows)
    return rows,{'game':int(g.game_id),'date':date,'home_team':home,'away_team':away,
                 'minutes':minutes,'source_shots':raw_n,'kept_shots':count,
                 'extra_nonparticipants_shift_labels':len(by_label)-len(label_to_side),
                 **{k:int(loses[k]) for k in ['missing_actor_or_team','no_consistent_boundary','counted_roster_disagrees']}}


def calendar(games):
    sides=pd.concat([games[['game_date','home_team_abbr']].rename(columns={'home_team_abbr':'team'}),
                     games[['game_date','away_team_abbr']].rename(columns={'away_team_abbr':'team'})],ignore_index=True)
    counts=pd.crosstab(sides.game_date,sides.team).sort_index().cumsum()
    assert counts.shape[1]==32
    return str(counts.index[counts.median(axis=1)>=20][0])


def prepare(end):
    season=(end-1)*10000+end
    sched=pd.read_parquet(RAW/f'nhl_schedule_{end}.parquet')
    sched=sched[sched.game_type.eq('R')&sched.game_state.isin(['OFF','FINAL'])].copy()
    if end==2027:
        sched=sched[sched.game_date.le(CUT)]
        official=pd.DataFrame(json.loads((OUT/'schedule.json').read_text()))
        assert set(sched.game_id)==set(official.id)
        ref=official.set_index('id')
        assert (sched.game_id.map(ref.homeTeam.map(lambda x:x['abbrev'])).map(team)==sched.home_team_abbr.map(team)).all()
        assert (sched.game_id.map(ref.awayTeam.map(lambda x:x['abbrev'])).map(team)==sched.away_team_abbr.map(team)).all()
        assert sched.game_id.map(ref.gameDate).eq(sched.game_date).all()
        cut=CUT
    else:
        assert len(sched)==1312
        cut=calendar(sched)
    assert sched.game_id.is_unique
    p=pd.read_parquet(RAW/f'play_by_play_lite_{end}.parquet',columns=PBP)
    s=pd.read_parquet(RAW/f'shifts_{end}.parquet',columns=['game_id','event_team','period','period_seconds','ids_on','ids_off'])
    assert not p[p.game_id.isin(sched.game_id)].duplicated(['game_id','event_id']).any()
    pg=p.groupby('game_id',sort=False)
    sg=s.groupby('game_id',sort=False)
    intervals=[]
    audits=[]
    for g in sched.sort_values(['game_date','game_id']).itertuples(index=False):
        q,a=prep_game(g,pg.get_group(g.game_id),sg.get_group(g.game_id),season)
        intervals.extend(q)
        audits.append(a)
    d=pd.DataFrame(intervals,columns=INTERVAL_COLUMNS)
    a=pd.DataFrame(audits)
    assert d.game.nunique()==len(sched) and (d.seconds>0).all()
    assert d.groupby('game').seconds.sum().le(3600).all()
    assert a.kept_shots.sum()==d[['home_shots','away_shots']].to_numpy().sum()
    assert a.kept_shots.sum()/a.source_shots.sum()>=.90,(season,a.sum(numeric_only=True).to_dict())
    assert a.minutes.between(15,60).all(),a[~a.minutes.between(15,60)]
    a['phase']=np.where(a.date.le(cut),'early','later')
    a.to_parquet(WORK/f'games_{season}.parquet',index=False)
    d.to_parquet(WORK/f'intervals_{season}.parquet',index=False)
    info={'season':season,'cutoff':cut,'games':len(a),'intervals':len(d),'kept_shots':int(a.kept_shots.sum()),
          'source_shots':int(a.source_shots.sum()),'recorded_lineup_diff':int(a.counted_roster_disagrees.sum()),
          'extra_nonparticipants_shift_labels':int(a.extra_nonparticipants_shift_labels.sum()),
          'median_eligible_minutes_per_game':float(a.minutes.median()),
          'phase_games':a.phase.value_counts().to_dict()}
    print(json.dumps(info),flush=True)
    return info


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((OUT/'bulk_manifest.json').read_text())['assets']
    for name,info in manifest.items():
        assert sha(RAW/name)==info['sha256'], name
    import sys
    years=[int(y) for y in sys.argv[1:]] or list(ENDS)
    reports=[prepare(y) for y in years]
    (OUT/('coverage.json' if years==list(ENDS) else f'coverage_{years[0]}.json')).write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    main()
