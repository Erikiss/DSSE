"""Native offline route and statistical controls for Gemma persistence."""
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch

import r1b_g_adapter as A
import r1b_g_k as K
import r1b_g_k_persistence as P
from r1b_g_k_test import MiniCtx


torch.set_num_threads(1)


class PersistenceTests(unittest.TestCase):
    def test_native_replayed_historical_edits_equal_incremental_cache(self):
        with tempfile.TemporaryDirectory() as td,patch('socket.socket.connect',side_effect=AssertionError('network')):
            for name in ('gemma4-e2b','gemma4-12b-qat'):
                ctx=MiniCtx(td,name,{},A.load_model(name,mock=True))
                E=K._engine(ctx);seq=E.encode('The river flows through town. The trees grow.',chat=False)
                pairs=[[0,2],[2,1]];positions=[5,6,7]
                out=P.cache_equivalence(E,seq,pairs,3.,positions)
                self.assertTrue(out['pass_'],out)
                clean=E.next_logits(seq)
                changed=E.next_logits(seq,setup=lambda:P._edit(E,pairs,'scale',3.,positions))
                self.assertGreater(float((clean-changed).abs().max()),0.)

    def test_forced_chain_consumes_the_same_absolute_crn_draws(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=MiniCtx(td,'chain',{},A.load_model('gemma4-e2b',mock=True));E=K._engine(ctx)
            prompt=E.encode('Add 1 and 2.');h=E.hidden([prompt],[1])[0][1][-1]
            cell=dict(seed=E.encode('3',chat=False)[0],block=1,state=h,prior=torch.zeros_like(h),
                      item=dict(id='x',prompt='Add 1 and 2.',P=[1,2],I=[3],Iplus=[],A=3),stratum='correct')
            refs=dict(c_mu=torch.zeros(E.layout.total),c_sd=torch.ones(E.layout.total))
            frame=E.encode('The river flows past trees.',chat=False)[:16]+E.encode('.',chat=False)
            pops={'H':[[1,3]]};unit=('x',0,0)
            first=P.chain(ctx,E,unit,cell,frame,pops,refs)
            replay=P.chain(ctx,E,unit,cell,frame,pops,refs,forced=first['tok'][1:8])
            noop=P.chain(ctx,E,unit,cell,frame,pops,refs,
                         op=dict(pairs=pops['H'],kind='scale',value=1.,streams='expert'))
            self.assertEqual(first['tok'],replay['tok'])
            self.assertEqual(first['tok'],noop['tok'])
            self.assertEqual(len(first['tok']),24)
            torch.testing.assert_close(first['aX']['H'],replay['aX']['H'],rtol=0,atol=0)

    def test_native_session_token_route_preserves_clean_twin_for_noop(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=MiniCtx(td,'session',{},A.load_model('gemma4-e2b',mock=True));E=K._engine(ctx)
            items={str(i):dict(id=str(i),prompt=f'Add {i} and 2.',A=i+2) for i in range(8)}
            template=dict(sid='s0',template=0,seed=0,items=list(items))
            pops={'H':[[1,3]]};refs=dict(c_mu=torch.zeros(E.layout.total),c_sd=torch.ones(E.layout.total))
            det=dict(beta=np.array([1.]),b=0.)
            clean=P.session(ctx,E,template,items,'ACH',[],1.,None,pops,refs,det)
            token=P.session(ctx,E,template,items,'ACH',pops['H'],1.,'TOKEN',pops,refs,det,
                            forced_first=clean['turns'][0]['tok'])
            kv=P.session(ctx,E,template,items,'ACH',pops['H'],1.,'KV',pops,refs,det,
                         forced_first=clean['turns'][0]['tok'])
            self.assertEqual([r['tok'] for r in clean['turns']],[r['tok'] for r in token['turns']])
            self.assertEqual([r['tok'] for r in clean['turns']],[r['tok'] for r in kv['turns']])
            self.assertFalse(token['historical_edit_positions'])
            self.assertEqual(len(kv['historical_edit_positions']),len(clean['turns'][0]['tok']))

    def test_tqa_schedule_respects_three_session_cap(self):
        items=[dict(qid=str(i)) for i in range(512)]
        templates=P._tqa_templates(items,192,False)
        self.assertEqual(len(templates),192)
        self.assertTrue(all(len(t)==8 and len(set(t))==8 for t in templates))
        self.assertEqual(set(Counter(x for t in templates for x in t).values()),{3})

    def test_carry_requires_a_resolved_denominator_and_specificity(self):
        r=P._ratio(np.arange(1.,21.)*2,np.arange(1.,21.),np.arange(20),200)
        self.assertTrue(r['defined']);self.assertEqual(r['K'],2.)
        row=dict(carry=r,p_pi=1.)
        self.assertIn('generic persistence',P._carry_string(row,dict(reject=False),True,'TOK'))
        undefined=P._ratio(np.ones(20),np.zeros(20),np.arange(20),200)
        self.assertFalse(undefined['defined'])
        self.assertIn('not defined',P._carry_string(dict(carry=undefined,p_pi=1.),dict(reject=False),True,'TOK'))


if __name__=='__main__':unittest.main(verbosity=2)
