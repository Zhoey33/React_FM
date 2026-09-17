"""Local-repair outcomes separated from later task-score changes. No API calls."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
if __package__:
    from .memory_direct_evidence import rows, mean, cluster_interval
else:
    from memory_direct_evidence import rows, mean, cluster_interval

ROOT = Path(__file__).resolve().parents[1]


def actions(row):
    return [s for s in row['steps'] if not s['action'].lower().startswith(('think:', 'think '))]


def repaired(cid, step):
    """Observable task-specific predicates; no LLM judge or post-score threshold."""
    obs, action = step['observation'].lower(), step['action'].lower()
    rooms = {'power-component__11__1':'hallway', 'chemistry-mix__17__1':'kitchen',
             'power-component__16__1':'foundry', 'melt__21__1':'kitchen', 'chemistry-mix__25__1':'hallway'}
    if cid in rooms:
        return obs.strip() in [f'you move to the {rooms[cid]}.', f'you move through the door to the {rooms[cid]}.']
    if cid == 'power-component__10__7':
        return obs.startswith('a wood cup') and 'blue paint' in obs
    if cid == 'melt__14__15':
        return obs.strip() == 'the freezer is now open.'
    if cid == 'chemistry-mix__16__8':
        return 'mix to produce smores' in obs
    if cid == 'chemistry-mix__16__14':
        return 'you move the orange to the kitchen.' in obs
    if cid == 'chemistry-mix__17__6':
        return obs.startswith('the recipe reads:')
    if cid == 'test-conductivity__450__31':
        return 'you move the paper clip to the green box.' in obs
    if cid == 'power-component__15__11':
        return 'electric motor' in obs and 'is now connected to' in obs
    if cid == 'boil__21__7':
        return 'drawer is now open' in obs
    if cid == 'melt__22__1':
        return 'inside the drawer is:' in obs
    if cid == 'grow-plant__93__27':
        return 'you move the orange seed to the inventory.' in obs
    if cid == 'grow-plant__94__21':
        return 'you focus on the orange seed.' in obs
    if cid == 'test-conductivity__675__23':
        return 'you move the sodium chloride to the green box.' in obs
    if cid == 'test-conductivity__676__14':
        return action.startswith(('look at glass jar', 'look in glass jar')) and 'sodium chloride' in obs
    raise ValueError('No predefined local criterion: ' + cid)


def summarize(branches, arms, repeats):
    groups = defaultdict(dict)
    for r in branches:
        groups[r['checkpoint_id']][(r['arm'], r['replay'])] = r
    expected = {(a, i) for a in arms for i in range(repeats)}
    complete = {c: d for c, d in groups.items() if set(d) == expected}
    detailed = []
    for cid, data in complete.items():
        for row in data.values():
            env_steps = actions(row)
            index = next((i for i, s in enumerate(env_steps, 1) if repaired(cid, s)), None)
            detailed.append({'checkpoint_id':cid, 'episode_id':row['episode_id'], 'arm':row['arm'],
                'replay':row['replay'], 'local_repair_action':index,
                'repaired_by_3':index is not None and index <= 3,
                'repaired_by_6':index is not None and index <= 6,
                'score_delta':row['score_after']-row['score_before'],
                'score_delta_at_3':env_steps[min(2,len(env_steps)-1)]['score']-row['score_before'] if env_steps else 0,
                'parse_errors_first_6':sum(s['observation'].startswith(('No known action', 'Unknown action')) for s in env_steps[:6]),
                'cost_cny':row['cost_cny'], 'memory_id':row['memory']['memory_id'],
                'status':row['status']})
    result = {'complete_checkpoints':len(complete), 'independent_episodes':len({r['episode_id'] for r in detailed}),
              'arms':{}, 'paired_local_repair_differences':{}, 'checkpoints':{}, 'rows':detailed}
    for arm in arms:
        rs=[r for r in detailed if r['arm']==arm]
        result['arms'][arm]={'branches':len(rs), 'repaired_by_3':sum(r['repaired_by_3'] for r in rs),
            'repaired_by_6':sum(r['repaired_by_6'] for r in rs),
            'mean_score_delta':mean([r['score_delta'] for r in rs]),
            'mean_score_delta_at_3':mean([r['score_delta_at_3'] for r in rs]),
            'mean_parse_errors_first_6':mean([r['parse_errors_first_6'] for r in rs]),
            'mean_cost_cny':mean([r['cost_cny'] for r in rs])}
    for cid in complete:
        result['checkpoints'][cid]={a:{'by3':sum(r['repaired_by_3'] for r in detailed if r['checkpoint_id']==cid and r['arm']==a),
            'by6':sum(r['repaired_by_6'] for r in detailed if r['checkpoint_id']==cid and r['arm']==a),
            'score_delta':mean([r['score_delta'] for r in detailed if r['checkpoint_id']==cid and r['arm']==a])} for a in arms}
    for arm in arms[1:]:
        ds=[(data[arm]['by3']-data['direct']['by3'])/repeats for data in result['checkpoints'].values()]
        clusters=[next(iter(complete[c].values()))['episode_id'] for c in result['checkpoints']]
        result['paired_local_repair_differences'][arm+'_minus_direct']={
            'mean_rate_difference':mean(ds), 'episode_cluster_bootstrap_95pct':cluster_interval(ds,clusters)}
    return result


def completed_test_rows(label):
    p=ROOT/'results'/('20260916_memory_increment_test_'+label)
    q=ROOT/'results'/('20260917_memory_increment_resume_'+label)
    original=rows(p/'branches.jsonl'); expected={(a,i) for a in ['direct','memory'] for i in range(2)}
    ids={r['checkpoint_id'] for r in original}
    complete={cid for cid in ids if {(r['arm'],r['replay']) for r in original if r['checkpoint_id']==cid}==expected}
    return [r for r in original if r['checkpoint_id'] in complete]+rows(q/'branches.jsonl')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path);args=p.parse_args()
    if args.diagnostic:
        result=summarize(rows(args.diagnostic/'branches.jsonl'),['direct','memory','curated'],3)
    else:
        result={b:summarize(completed_test_rows(b),['direct','memory'],2) for b in ['b006','b012']}
    print(json.dumps(result,ensure_ascii=False,indent=2))
