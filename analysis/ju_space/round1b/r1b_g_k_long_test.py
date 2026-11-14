"""Offline numeric alignment and actual native-generation tests for G K8c."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import r1b_g_adapter as A
import r1b_g_k as K
import r1b_g_k_long as L
import r1b_k_labels as KL
from r1b_g_k_test import MiniCtx
from r1b_k_s4 import parse_long


torch.set_num_threads(1)


class LongTests(unittest.TestCase):
    def test_value_spans_exclude_step_numbers_and_preserve_missing_values(self):
        tok,_=A.load_model('gemma4-e2b',mock=True)
        item=dict(start=10,steps=[dict(k=1,op='add',b=2,result=12,cls='a'),
             dict(k=2,op='subtract',b=15,result=-3,cls='b'),dict(k=3,op='add',b=1,result=-2,cls='c')])
        text='Step 1: 12\nStep 2: -3\nNo result for the last step.'
        tokens=tok(text,add_special_tokens=False)['input_ids']
        a=L.value_spans(tok,tokens,item)
        self.assertEqual([tok.decode([tokens[i] for i in ix]) for ix in a['value_positions']],['12','-3',''])
        self.assertEqual(a['newline_positions'][:2],[text.index('\n'),text.index('\n',text.index('\n')+1)])
        steps,fmt=parse_long(text,item)
        self.assertTrue(steps[0]['local']);self.assertTrue(steps[1]['local'])
        self.assertIsNone(steps[2]['v']);self.assertFalse(steps[2]['local']);self.assertFalse(fmt)

    def test_local_correctness_distinguishes_carried_wrong_value(self):
        item=dict(start=10,steps=[dict(k=1,op='add',b=2,result=12,cls='a'),dict(k=2,op='add',b=5,result=17,cls='b')])
        steps,_=parse_long('Step 1: 13\nStep 2: 18\n',item)
        self.assertFalse(steps[0]['local']);self.assertTrue(steps[1]['local']);self.assertFalse(steps[1]['glob'])

    def test_hazard_uses_previous_value_and_retains_omission_error(self):
        item=dict(id='x',steps=[dict(k=1,result=12,cls='a',op='add'),dict(k=2,result=17,cls='b',op='add')])
        row=dict(item='x',sample=0,fmt_ok=True,istar=1,steps=[{'local':True},{'local':False}],
            per_step=[dict(Ht=2.,rs=[1.,3.],rs_tilde=[-2.,2.],ent=.4,value_positions=[5,6],token_position=105),None],
            prompt_ids=[1]*100,tok=[2]*10,rep4=False,preamble=False)
        X,y,g,Rt,meta=L.hazard_table([row],{'x':item})
        self.assertEqual(y.tolist(),[1.]);self.assertEqual(X[0][0],2.)
        self.assertEqual(X[0][2],110);self.assertEqual(meta[0]['previous_value_positions'],[5,6])
        self.assertTrue(meta[0]['missing_current_value'])
        np.testing.assert_array_equal(Rt,[[-2.,2.]])

    def test_native_mock_does_not_fill_unparsed_answers_with_oracle_values(self):
        with tempfile.TemporaryDirectory() as td,patch('socket.socket.connect',side_effect=AssertionError('network')):
            model=A.load_model('gemma4-e2b',mock=True);ctx=MiniCtx(td,'k_long',{},model);E=K._engine(ctx)
            items=KL.gen_ach_long(n=1);hn={'RS39':[[[1,4]],[[1,5]],[[1,6]]]}
            refs={'c_mu':torch.zeros(E.layout.total),'c_sd':torch.ones(E.layout.total)}
            rows=L.collect(ctx,E,items,hn,refs,dict(beta=np.array([1.]),b=0.),[[1,3]])
            self.assertEqual(len(rows),2)
            for row in rows:
                steps,fmt=parse_long(row['text'],items[0])
                self.assertEqual(row['steps'],steps);self.assertEqual(row['fmt_ok'],fmt)
                self.assertEqual(row['istar'],next((i for i,s in enumerate(steps) if not s['local']),None))
                for index,step in enumerate(steps):
                    if step['v'] is None:self.assertFalse(row['value_positions'][index])
                self.assertTrue(all(0<=p['cett_share_H']<=1 for p in row['per']))
            result=L.analyze(ctx,{'rows':rows},{items[0]['id']:items[0]})
            self.assertEqual(result['n_first_errors'],0)
            self.assertEqual(result['G_K13'],'format not followed')


if __name__=='__main__':unittest.main(verbosity=2)
