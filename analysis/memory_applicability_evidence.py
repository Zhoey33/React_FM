"""Predeclared four-arm comparisons; checkpoint-level paired bootstrap."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.memory_direct_evidence import rows, mean, cluster_interval
from src.scienceworld_failure_detector import ScienceWorldFailureDetector

ARMS=['direct','observe','memory','checked']


def repaired(step, criterion):
    return bool(re.search(criterion['observation_regex'],step['observation'],re.I)) and (
        not criterion.get('action_regex') or bool(re.search(criterion['action_regex'],step['action'],re.I)))


def analyze(directory):
    bs=rows(directory/'branches.jsonl');criteria=json.loads((directory/'criteria.json').read_text())
    groups=defaultdict(dict)
    for r in bs:groups[r['checkpoint_id']][(r['arm'],r['replay'])]=r
    expected={(a,i) for a in ARMS for i in range(2)}
    complete={cid:d for cid,d in groups.items() if set(d)==expected}
    details=[];detector=ScienceWorldFailureDetector(enable_implicit_failures=False)
    for cid,data in complete.items():
        for r in data.values():
            steps=[s for s in r['steps'] if not s['action'].lower().startswith(('think:', 'think '))]
            success=next((i for i,s in enumerate(steps,1) if repaired(s,criteria[cid])),None)
            repair_steps=[s for s in steps if s['kind']!='inspection']
            detection=detector.detect(repair_steps[0]['observation'],repair_steps[0]['action'],[]) if repair_steps else None
            blocked=[]
            for s in steps:
                d=detector.detect(s['observation'],s['action'],[])
                blocked.append(bool(d.is_failure and d.failure_type!='ambiguity'))
            prep=r['preparation'] or {}
            details.append({'checkpoint_id':cid,'episode_id':r['episode_id'],'arm':r['arm'],'replay':r['replay'],
                'by3':success is not None and success<=3,'by6':success is not None and success<=6,
                'repair_action_index':success,'first_repair_blocked':bool(detection and detection.is_failure and detection.failure_type!='ambiguity'),
                'blocked_actions':sum(blocked),'inspection_blocked':any(flag and s['kind']=='inspection' for flag,s in zip(blocked,steps)),
                'score_delta':r['score_after']-r['score_before'],'reached100':r['score_after']>=100,
                'history_hit':r['memory']['memory_id'] is not None,'history_accepted':prep.get('use_memory',False),
                'inspection':prep.get('inspection','NONE'),'status':r['status'],'cost_cny':r['cost_cny'],
                'shared_from':r['shared_from']})
    result={'complete_checkpoints':len(complete),'complete_episodes':len({r['episode_id'] for r in details}),
            'rows':details,'arms':{},'contrasts':{},'hit_only_contrasts':{},'per_checkpoint':{}}
    for a in ARMS:
        rs=[r for r in details if r['arm']==a]
        result['arms'][a]={'branches':len(rs),'by3':sum(r['by3'] for r in rs),'by6':sum(r['by6'] for r in rs),
            'first_repair_blocked':sum(r['first_repair_blocked'] for r in rs),
            'blocked_actions':sum(r['blocked_actions'] for r in rs),'inspection_blocked':sum(r['inspection_blocked'] for r in rs),
            'mean_score_delta':mean([r['score_delta'] for r in rs]),'reached100':sum(r['reached100'] for r in rs),
            'inspections':sum(r['inspection']!='NONE' for r in rs),
            'history_accepted':sum(r['history_accepted'] for r in rs),'mean_counterfactual_cost_cny':mean([r['cost_cny'] for r in rs])}
    for cid in complete:
        result['per_checkpoint'][cid]={a:{'by3':sum(r['by3'] for r in details if r['checkpoint_id']==cid and r['arm']==a),
            'by6':sum(r['by6'] for r in details if r['checkpoint_id']==cid and r['arm']==a)} for a in ARMS}
    for hit_only,key in [(False,'contrasts'),(True,'hit_only_contrasts')]:
        for treatment,control in [('checked','observe'),('checked','memory'),('observe','direct'),('memory','direct')]:
            ds=[];clusters=[]
            for cid,data in complete.items():
                if hit_only and next(iter(data.values()))['memory']['memory_id'] is None:continue
                ds.append((result['per_checkpoint'][cid][treatment]['by3']-result['per_checkpoint'][cid][control]['by3'])/2)
                clusters.append(next(iter(data.values()))['episode_id'])
            result[key][treatment+'_minus_'+control]={'checkpoints':len(ds),'by3_rate_difference':mean(ds),
                'episode_bootstrap_95pct':cluster_interval(ds,clusters)}
    calls={r['call_id']:r for r in rows(directory/'calls.jsonl')}
    result.update(physical_calls=len(calls),unsettled_calls=sum(r['status']!='complete' for r in calls.values()),
                  physical_cost_cny=sum(r['accounted_cny'] for r in calls.values()),
                  history_hit_checkpoints=sum(next(iter(d.values()))['memory']['memory_id'] is not None for d in complete.values()),
                  shared_rows=sum(r['shared_from'] is not None for r in details))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(analyze(args.directory),ensure_ascii=False,indent=2))
