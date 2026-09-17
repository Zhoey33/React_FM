"""Four-arm applicability diagnostic; all calls share an explicit run budget."""
import argparse
import sys
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.compare_memory_direct import (
    ROOT, TASKS, Calls, BudgetExhausted, ScienceWorldEnv, actor_instructions,
    actor_prompt, snapshot, restore, execute, normalized, read_rows, append, action_from,
)

ARMS = ['direct', 'observe', 'memory', 'checked']
CHECKER_VERSION = 2
CUE = 'The previous action failed. Reconsider your approach using the observed state.'
PREPARATION_SYSTEM = '''You select one optional inspection before a separate agent repairs a failure.
Return JSON only with keys inspection, use_memory, evidence_quote, reason.
inspection: NONE, look around, inventory, look at OBJ, look in OBJ, or read OBJ.
Use observed object referents only. Do not open, move, connect, focus, or manipulate objects.
If a numbered disambiguation menu is pending, inspection must be NONE.
If no historical experience is supplied, use_memory must be false.
If history is supplied, use_memory may be true ONLY when current observations explicitly
support the same target/mechanism and the prerequisites of that historical repair.
Do not infer that a container is empty/open or that objects have matching terminals.
For true, inspection must be NONE and evidence_quote must be an exact excerpt from
current or recent environment observations supporting applicability.
If a condition is contradicted or unknown, use_memory must be false; choose a useful
inspection if possible, otherwise NONE. Never invent evidence.
Keep reason under 30 words. A separate fresh repairer will receive only actual observations
and, when accepted, the unchanged history. Your reasoning will not be passed to it.'''


def canonical_state(state: dict) -> dict:
    return dict(state, look=normalized(state['look']))


def preparation_request(cp: dict, memory: dict) -> str:
    history = [(s['action'], s['observation']) for s in cp['steps']]
    prompt = actor_prompt(cp['task'], cp['initial'], history, canonical_state(cp['state']), CUE)
    if memory['text']:
        prompt += '\nHistorical experience under review:\n' + memory['text']
    return prompt + '\nSelect the inspection and applicability decision as JSON, not a repair action.'


def interpret_preparation(raw: str, cp: dict, memory: dict) -> dict:
    try:
        value = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
        if not isinstance(value, dict):
            raise ValueError('not an object')
    except (ValueError, TypeError):
        return {'inspection':'NONE', 'use_memory':False, 'reason':'invalid_json', 'raw':raw}
    inspection = str(value.get('inspection', 'NONE')).strip()
    if not re.fullmatch(r'NONE|look around|inventory|(?:look at|look in|read) [^\n]{1,160}', inspection):
        inspection = 'NONE'
    pending = cp['steps'][-1]['observation'].startswith('Ambiguous request:')
    if pending:
        inspection = 'NONE'
    evidence = str(value.get('evidence_quote', '')).strip()
    observed = '\n'.join([cp['state']['look'], cp['state']['inventory']] +
                         [s['observation'] for s in cp['steps'][-10:]])
    accepted = (value.get('use_memory') is True and bool(memory['text']) and
                str(value.get('inspection', '')).strip() == 'NONE' and
                len(evidence) >= 8 and evidence in observed)
    reason_code = ''
    # A real switch quote must not justify terminal claims about a motor.
    # Limit this check to the observed connection-target failure, not all objects.
    failed_action = cp['steps'][-1]['action'].lower()
    if accepted and failed_action.startswith('connect ') and ' to ' in failed_action:
        target = failed_action.split(' to ', 1)[1]
        target = re.sub(r'\s+(?:terminal\s+[12]|anode|cathode)$', '', target).strip()
        quote = evidence.lower().splitlines()[0].strip()
        port = r'(?:anode|cathode|terminal\s+[12])'
        subject_evidence = (re.match(r'(?:a |an |the )?' + re.escape(target) + r'\b', quote)
                            and re.search(r'\bits\s+' + port, quote))
        relational_evidence = re.search(port + r'\s+(?:on|of)\s+(?:the\s+)?' + re.escape(target) + r'\b', quote)
        if not (subject_evidence or relational_evidence):
            accepted = False
            reason_code = 'connection_target_not_supported_by_quote'
    return {'inspection':inspection, 'use_memory':accepted, 'evidence_quote':evidence,
            'reason':str(value.get('reason', '')), 'reason_code':reason_code, 'raw':raw}


def repair_request(cp: dict, history: list, state: dict, memory_text: str) -> str:
    prompt = actor_prompt(cp['task'], cp['initial'], history, canonical_state(state), CUE)
    if memory_text:
        prompt += '\nHistorical experience, to use only if applicable:\n' + memory_text
    return prompt + '\nDiagnose this failure and suggest a repair.'


def run_one(cp: dict, arm: str, replay: int, memory: dict, env, calls, output: Path) -> dict:
    group = f"branch/{cp['checkpoint_id']}/{replay}/{arm}"
    restore(env, cp)
    history = [(s['action'], s['observation']) for s in cp['steps']]
    score = cp['state']['score']; state = cp['state']; steps = []; preparation = None
    advice = ''; status = 'actor_horizon'; accepted_text = ''
    try:
        if arm in ['observe', 'checked']:
            reviewed = memory if arm == 'checked' else {'text':''}
            raw = calls.generate(preparation_request(cp, reviewed), group, .1, repair=True,
                                 system_override=PREPARATION_SYSTEM)
            preparation = interpret_preparation(raw, cp, reviewed)
            if preparation['inspection'] != 'NONE':
                action = preparation['inspection']; obs, score, done = execute(env, action, score)
                history.append((action, obs)); state = snapshot(env, score)
                steps.append({'action':action,'observation':obs,'score':score,'kind':'inspection'})
                if done:
                    status = 'terminal'
                    return result_row(cp, arm, replay, group, memory, preparation, advice, steps, score, status, calls)
            if arm == 'checked' and preparation['use_memory']:
                accepted_text = memory['text']
        elif arm == 'memory':
            accepted_text = memory['text']
        advice = calls.generate(repair_request(cp, history, state, accepted_text), group, .1, repair=True)
        pending = True
        for turn in range(18):
            guidance = CUE + '\nCurrent-state repair advice:\n' + advice if pending else ''
            # Before the first new observation, all arms use the saved checkpoint state.
            prompt = actor_prompt(cp['task'], cp['initial'], history, canonical_state(state), guidance)
            action = action_from(calls.generate(prompt, group, .1))
            if not action:
                raise RuntimeError('Empty actor response')
            obs, score, done = execute(env, action, score)
            is_action = not action.lower().startswith(('think:', 'think '))
            if is_action:
                pending = False
                state = snapshot(env, score)
            history.append((action, obs))
            steps.append({'action':action,'observation':obs,'score':score,'kind':'actor'})
            if done:
                status = 'terminal'; break
            if sum(not s['action'].lower().startswith(('think:', 'think ')) for s in steps) >= 6:
                status = 'action_horizon'; break
    except BudgetExhausted:
        status = 'budget'
    return result_row(cp, arm, replay, group, memory, preparation, advice, steps, score, status, calls)


def result_row(cp, arm, replay, group, memory, preparation, advice, steps, score, status, calls):
    return {'group':group,'checkpoint_id':cp['checkpoint_id'],'episode_id':cp['episode_id'],
            'task':cp['task'],'arm':arm,'replay':replay,'memory':memory,'preparation':preparation,
            'advice':advice,'steps':steps,'score_before':cp['state']['score'],'score_after':score,
            'status':status,'cost_cny':calls.spent(group),'shared_from':None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=p.parse_args(); output=args.output
    cps=read_rows(output/'checkpoints.jsonl');plan=json.loads((output/'retrieval_plan.json').read_text())
    assert len(cps)<=18
    assert (output/'criteria.json').exists(), 'Freeze local-repair criteria before running'
    protocol={'checker_version':CHECKER_VERSION,'model':'deepseek-ai/DeepSeek-V4-Flash','arms':ARMS,'replays':2,'split':'dev',
              'branch_cny':.1,'run_cny':16,'environment_actions':6,'actor_turns':18,
              'preparation_system':PREPARATION_SYSTEM,'shared_miss_trajectories':True}
    if (output/'protocol.json').exists():
        assert json.loads((output/'protocol.json').read_text())==protocol
    else:
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    calls=Calls(output,protocol['model'],16,True,True)
    completed={r['group'] for r in read_rows(output/'branches.jsonl')}
    started={r['group'] for r in calls.calls.values()}
    if started-completed:
        raise RuntimeError('Interrupted groups exist; preserve them and restart complete paired blocks in a new directory')

    def worker(partition):
        env=ScienceWorldEnv(task_names=TASKS,split='dev',max_variations_per_task=5,env_step_limit=100)
        try:
            env.setup()
            for index,cp in enumerate(cps):
                if index%args.workers!=partition: continue
                env.env.load(cp['task'],0)
                assert cp['variation'] in env.env.get_variations_dev()
                assert cp['variation'] not in env.env.get_variations_train()
                memory=plan[cp['checkpoint_id']]['automatic'];miss=memory['memory_id'] is None
                for replay in range(2):
                    arms=['direct','observe'] if miss else list(ARMS)
                    random.Random(20260917+index*101+replay).shuffle(arms)
                    for arm in arms:
                        group=f"branch/{cp['checkpoint_id']}/{replay}/{arm}"
                        if group in completed: continue
                        row=run_one(cp,arm,replay,memory,env,calls,output)
                        append(output/'branches.jsonl',row)
                        if miss:
                            alias='memory' if arm=='direct' else 'checked'
                            append(output/'branches.jsonl',dict(row,arm=alias,
                                group=f"branch/{cp['checkpoint_id']}/{replay}/{alias}",shared_from=group))
                        print(json.dumps({'finished':group,'score':row['score_after'],'cost':round(calls.spent(),4)}),flush=True)
        finally:env.close()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(worker,i) for i in range(args.workers)]: future.result()
    print(json.dumps({'accounted_cny':calls.spent()}),flush=True)


if __name__=='__main__':main()
