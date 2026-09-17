import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.compare_memory_direct import *
from src.memory import FailureMemoryStore
p=Path('results/20260917_memory_diagnostic_dev')
specs={
 'power-component__10__7':(None,['0'],'No same-task record found for this menu; retain as bank-coverage control.'),
 'power-component__11__1':(127,['open door to hallway','go to hallway'],'Exact closed-door mechanism and destination.'),
 'melt__14__15':(None,['open freezer'],'No same-task record for removing the invented word door; coverage control.'),
 'chemistry-mix__16__8':(333,['move chocolate to glass cup','move marshmallow to glass cup','mix glass cup'],'Existing ingredient-to-container then mix rule, adapted only in validation to observed ingredients and empty cup.'),
 'chemistry-mix__16__14':(336,['0'],'Select current menu referent; historical numeric label is not transferable.'),
 'chemistry-mix__17__1':(322,['open door to kitchen','go to kitchen'],'Exact closed-door mechanism and destination.'),
 'chemistry-mix__17__6':(353,['read recipe'],'Generic recipe referent; existing record explicitly includes read recipe.'),
 'test-conductivity__450__31':(152,['move paper clip to green box'],'Placement grammar transfers to current object and box; does not validate the previous conductivity conclusion.')}
cps=[c for c in read_rows(Path('results/20260916_memory_vs_direct_v2/checkpoints.jsonl')) if c['checkpoint_id'] in specs]
store=FailureMemoryStore(scope='task_type',retrieval_mode='hybrid');store.load(str(MEMORY));entries={e.memory_id:e for e in store.get_bucket_entries(cross_env=True)}
env=ScienceWorldEnv(task_names=TASKS,split='dev',max_variations_per_task=2,env_step_limit=100);plan={};probes=[]
try:
 env.setup()
 for cp in cps:
  mid,actions,reason=specs[cp['checkpoint_id']]
  restore(env,cp);state=cp['state']['score'];trace=[]
  for action in actions:
   obs,state,done=execute(env,action,state);trace.append({'action':action,'observation':obs,'score':state,'done':done})
   if done:break
  if mid is None: curated={'memory_id':None,'text':'','rejection':'no_verified_same_task_record'}
  else:
   e=entries[mid];assert e.task_type==cp['task']
   curated={'memory_id':mid,'text':('Past failed action: '+e.failure_action+'\nPast observation: '+e.failure_observation+'\nHistorical repair: '+e.get_repair_display())[:1400],'rejection':None}
  plan[cp['checkpoint_id']]={'automatic':choose_memory(store,cp),'curated':curated,'selection_reason':reason}
  probes.append({'checkpoint_id':cp['checkpoint_id'],'actions_not_shown_to_model':trace,'selection_reason':reason})
  print(cp['checkpoint_id'],json.dumps(trace,ensure_ascii=False),flush=True)
finally:env.close()
(p/'checkpoints.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cps))
(p/'retrieval_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
(p/'validation_probes.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2))
