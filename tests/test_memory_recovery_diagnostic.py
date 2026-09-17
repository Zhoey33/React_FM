from analysis.memory_recovery_diagnostic import repaired, summarize


def test_local_success_requires_the_original_target():
    wrong = {'action': 'connect red wire terminal 2 to switch anode',
             'observation': 'terminal 2 on red wire is now connected to anode on switch'}
    right = {'action': 'connect red wire terminal 1 to electric motor anode',
             'observation': 'terminal 1 on red wire is now connected to anode on electric motor'}
    assert not repaired('power-component__15__11', wrong)
    assert repaired('power-component__15__11', right)
    for obs in ['You move to the kitchen.', 'You move through the door to the kitchen.']:
        assert repaired('melt__21__1', {'action':'go kitchen', 'observation':obs})
    assert not repaired('melt__21__1', {'action':'open door to kitchen',
                                      'observation':'The door is now open.'})


def test_thoughts_do_not_count_as_environment_actions_and_incomplete_pairs_excluded():
    base = {'checkpoint_id':'melt__21__1', 'episode_id':'melt__21',
            'score_before':0, 'score_after':70, 'cost_cny':.01,
            'memory':{'memory_id':None}, 'status':'action_horizon', 'replay':0,
            'steps':[{'action':'think: plan','observation':'OK.','score':0}]*5 +
                    [{'action':'open door to kitchen','observation':'The door is now open.','score':0},
                     {'action':'go kitchen','observation':'You move through the door to the kitchen.','score':0},
                     {'action':'focus on tin','observation':'You focus on tin.','score':70}]}
    assert summarize([dict(base,arm='direct')],['direct','memory'],1)['complete_checkpoints']==0
    result=summarize([dict(base,arm=a) for a in ['direct','memory']],['direct','memory'],1)
    assert all(r['local_repair_action']==2 for r in result['rows'])
    assert result['arms']['direct']['repaired_by_3']==1
    assert result['arms']['direct']['mean_score_delta']==70
