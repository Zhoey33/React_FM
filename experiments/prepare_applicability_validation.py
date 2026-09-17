"""Freeze new checkpoints, automatic retrieval and observable repair criteria before arms run."""
import json
import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.compare_memory_direct import (ROOT, TASKS, MEMORY, ScienceWorldEnv, FailureMemoryStore,
    read_rows, restore, execute, choose_memory)

SOURCE=ROOT/'results/20260917_applicability_collect'
OUTPUT=ROOT/'results/20260917_applicability_validation'

# Explicit offline positive controls, never supplied to the experimental models.
PROBES={
 'power-component__12__26':['connect black wire terminal 2 to electric motor anode'],
 'power-component__13__24':['connect red wire terminal 2 to electric motor anode'],
 'power-component__14__0':['open door to hallway','go hallway','open door to workshop','go workshop','focus on electric motor'],
 'boil__16__23':['pick up soap','0'],
 'melt__16__6':['0'],
 'melt__17__38':['open blast furnace'],
 'boil__18__35':['disconnect green wire'],
 'chemistry-mix__20__19':['go hallway','open door to kitchen','go kitchen'],
 'grow-plant__64__10':['0'],
 'grow-plant__65__20':['0'],
 'grow-plant__66__9':['0'],
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    cps=read_rows(SOURCE/'checkpoints.jsonl')
    if args.freeze:
        assert len(read_rows(SOURCE/'episodes.jsonl'))==18
        assert all(c['checkpoint_id'] in PROBES for c in cps)
    env=ScienceWorldEnv(task_names=TASKS,split='dev',max_variations_per_task=5,env_step_limit=100)
    results=[]
    try:
        env.setup()
        for cp in cps:
            if cp['checkpoint_id'] not in PROBES:continue
            restore(env,cp);score=cp['state']['score'];trace=[]
            for action in PROBES[cp['checkpoint_id']]:
                obs,score,done=execute(env,action,score)
                trace.append({'action':action,'observation':obs,'score':score})
                if done:break
            results.append({'checkpoint_id':cp['checkpoint_id'],'trace':trace})
            print(cp['checkpoint_id'],json.dumps(trace),flush=True)
    finally:env.close()
    (OUTPUT/'positive_controls.json').write_text(json.dumps(results,indent=2))
    if args.freeze:
        from analysis.memory_applicability_evidence import repaired
        criteria=json.loads((OUTPUT/'criteria.json').read_text())
        assert set(criteria)=={c['checkpoint_id'] for c in cps}
        assert all(any(repaired(s,criteria[r['checkpoint_id']]) for s in r['trace']) for r in results)
        old=[]
        for name in ['20260916_memory_vs_direct_v2','20260916_memory_increment_test_b012']:
            old += read_rows(ROOT/'results'/name/'checkpoints.jsonl')
        assert not {(c['task'],c['variation']) for c in cps} & {(c['task'],c['variation']) for c in old}
        store=FailureMemoryStore(scope='task_type',retrieval_mode='hybrid');store.load(str(MEMORY))
        plan={c['checkpoint_id']:{'automatic':choose_memory(store,c)} for c in cps}
        (OUTPUT/'checkpoints.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cps))
        (OUTPUT/'retrieval_plan.json').write_text(json.dumps(plan,indent=2))
        (OUTPUT/'collection_source.json').write_text(json.dumps({'source':str(SOURCE),'separate_collection_cost':True},indent=2))
        print('Frozen checkpoints',len(cps),'memory hits',sum(v['automatic']['memory_id'] is not None for v in plan.values()),flush=True)


if __name__=='__main__':main()
