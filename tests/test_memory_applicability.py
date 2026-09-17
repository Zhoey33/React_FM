import json
from unittest.mock import patch
from experiments import validate_memory_applicability as v

CP = {'checkpoint_id':'case','episode_id':'episode','task':'melt','initial':'Melt tin',
      'state':{'score':0,'look':'The glass cup is empty.','inventory':'empty'},
      'steps':[{'action':'mix tin','observation':'No known action matches that input.','score':0}]}
MEMORY = {'memory_id':1,'text':'UNIQUE HISTORICAL CONTENT'}


def test_gate_needs_observed_evidence_and_no_simultaneous_inspection():
    raw={'inspection':'NONE','use_memory':True,'evidence_quote':'The glass cup is empty.'}
    assert v.interpret_preparation(json.dumps(raw),CP,MEMORY)['use_memory']
    assert not v.interpret_preparation(json.dumps(dict(raw,evidence_quote='The bowl is empty.')),CP,MEMORY)['use_memory']
    assert not v.interpret_preparation(json.dumps(dict(raw,inspection='look at bowl')),CP,MEMORY)['use_memory']
    assert not v.interpret_preparation(json.dumps(raw),CP,{'text':''})['use_memory']
    assert not v.interpret_preparation('invalid JSON',CP,MEMORY)['use_memory']


def test_pending_menu_and_disallowed_manipulation_are_not_inspections():
    cp=dict(CP,steps=[{'action':'open drawer','observation':'Ambiguous request: choose a number','score':0}])
    assert v.interpret_preparation('{"inspection":"look around"}',cp,MEMORY)['inspection']=='NONE'
    assert v.interpret_preparation('{"inspection":"open freezer"}',CP,MEMORY)['inspection']=='NONE'


def test_switch_evidence_does_not_support_motor_terminals_but_motor_evidence_does():
    switch='a switch, which is off. its anode is connected to: nothing. its cathode is connected to: nothing.'
    motor='a electric motor, which is off. its anode is connected to: nothing. its cathode is connected to: nothing.'
    cp=dict(CP,state=dict(CP['state'],inventory=switch+'\n'+motor),steps=[{
        'action':'connect black wire terminal 2 to electric motor terminal 1',
        'observation':'No known action matches that input.','score':0}])
    decision={'inspection':'NONE','use_memory':True,'evidence_quote':switch}
    rejected=v.interpret_preparation(json.dumps(decision),cp,MEMORY)
    assert not rejected['use_memory']
    assert rejected['reason_code']=='connection_target_not_supported_by_quote'
    decision['evidence_quote']=motor
    assert v.interpret_preparation(json.dumps(decision),cp,MEMORY)['use_memory']


def test_rejected_history_and_preparation_reason_do_not_leak_to_repair(tmp_path):
    captured=[]
    class Calls:
        def generate(self,prompt,*args,repair=False,system_override=None):
            captured.append((prompt,repair,system_override))
            if system_override:
                return json.dumps({'inspection':'look at glass cup','use_memory':False,
                                   'reason':'UNIQUE REJECTED REASON'})
            return 'wait'
        def spent(self,*args):return .01
    with patch.object(v,'restore'),patch.object(v,'snapshot',return_value=CP['state']),\
         patch.object(v,'execute',return_value=('The glass cup is empty.',0,False)):
        row=v.run_one(CP,'checked',0,MEMORY,object(),Calls(),tmp_path)
    assert MEMORY['text'] in captured[0][0]
    for prompt,_,_ in captured[1:]:
        assert MEMORY['text'] not in prompt
        assert 'UNIQUE REJECTED REASON' not in prompt
    assert row['steps'][0]['kind']=='inspection'
    assert len(row['steps'])==6  # inspection consumes one of the six actions
    assert row['status']=='action_horizon'
