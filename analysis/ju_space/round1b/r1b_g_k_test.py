"""CPU tests for independent Gemma K; no real models or network access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import r1b_core as RC
import r1b_g_adapter as GA
import r1b_g_k as GK

torch.set_num_threads(1)


class MiniCtx:
    mock=True
    device="cpu"
    freeze={}
    model_name="gemma4-e2b"
    def __init__(self,root,stage,data,model):
        self.root=str(root);self.stage=stage;self.work=str(Path(root)/stage)
        self._data=data;self._model=model;Path(self.work).mkdir(exist_ok=True)
    def data(self):return self._data
    def model(self,name=None):return self._model
    def path(self,rel):
        p=Path(self.work)/rel;p.parent.mkdir(exist_ok=True,parents=True);return str(p)
    def artifact(self,stage,rel):return str(Path(self.root)/stage/rel)
    def write_json(self,rel,obj,**kwargs):RC.dump_json(obj,self.path(rel))
    def save_pt(self,rel,obj,**kwargs):torch.save(obj,self.path(rel))
    def key(self,*args):return RC.keyseed(*args)
    def cached(self,key,fn):
        p=self.path('cache/'+key+'.pt')
        if Path(p).exists():return torch.load(p,weights_only=False)
        obj=fn();torch.save(obj,p);return obj


def fixture_data():
    tqa=[]
    for split,n in [('S1',48),('S2',4),('S3',12),('S4a',4)]:
        for i in range(n):
            tqa.append(dict(qid=f'{split}_{i}',question=f'Which digit is {i % 10}?',aliases=[str(i % 10)],split=split))
    ar=[]
    for split,n in [('A-scr',16),('A-csel',4),('A-det',8),('A-bat',4),('A-sy',4)]:
        for i in range(n):
            ar.append(dict(id=f'{split}_{i}',prompt=f'Add {i%5} and 2.',A=i%5+2,split=split))
    txt='The river passes the town. The tree grows near the river. '
    return dict(tqa=tqa,ach_short=ar,ne=[dict(id=f'ne{i}',prompt=f'Who founded imaginary city {i}?') for i in range(8)],
                wt=dict(train=[txt*12],refw=[txt*3,txt[::-1]*3],ntext=[txt*3]*4),ref_k16=[txt],
                ref_r16_prompts=['Name a tree.','Name a river.'],
                fp=[dict(id=f'fp{i}',prompt=f'Please say {i+3}.',comply=i+3) for i in range(4)],
                mc=[dict(id=f'mc{i}',prompt=f'The invented name is City{i}. Repeat it.',planted=f'City{i}') for i in range(4)])


class GemmaKTests(unittest.TestCase):
    def test_allocation_is_disjoint_and_shrinks_L_first(self):
        s=GK._sizes(type('C',(),{'mock':True})())
        rows=[dict(cls=c,span=2,row={'qid':f'{c}{i}'}) for c in ('CC','CI') for i in range(5)]
        out,yield_=GK._allocate(rows,s)
        ids=[r['row']['qid'] for r in out]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(yield_['allocated']['S_fit'],[4,4])
        self.assertEqual(yield_['allocated']['S_L'],[0,0])

    def test_ragged_controls_match_layer_counts_and_exclude_H(self):
        layout=GA.NeuronLayout((8,16,5))
        h=[[0,2],[1,3],[1,7],[2,1]]
        out=GK._candidates(layout,h,torch.ones(layout.total),5,max_draws=20)
        self.assertEqual(len(out),5)
        for c in out:
            self.assertEqual(sorted(l for l,j in c['neurons']),[0,1,1,2])
            self.assertFalse(set(map(tuple,h)) & set(map(tuple,c['neurons'])))
            for l,j in c['neurons']:self.assertLess(j,layout.widths[l])

    def test_duplicate_raw_items_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            GK._check_disjoint({'tqa':[{'qid':'x','split':'S1'},{'qid':'x','split':'S3'}]})

    def test_exact_reference_tail_resumes_without_statistical_change(self):
        with tempfile.TemporaryDirectory() as td:
            model=GA.load_model('gemma4-e2b',mock=True)
            ctx=MiniCtx(td,'k_prepare',fixture_data(),model)
            E=GK._engine(ctx)
            expected=GK._reference_stats(ctx,E,ctx.data())
            ctx.time_left=lambda:0
            with self.assertRaises(RC.StepYield):GK._reference_stats(ctx,E,ctx.data())
            del ctx.time_left
            actual=GK._reference_stats(ctx,E,ctx.data())
            for name in ('p998_pos','p998_neg','z_mu','c_mu'):
                torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)

    def test_empty_continuation_is_retained_without_inventing_answer_features(self):
        with tempfile.TemporaryDirectory() as td:
            ctx=MiniCtx(td,'k_measure',{},GA.load_model('gemma4-e2b',mock=True));E=GK._engine(ctx)
            rows=[dict(qid=f'q{i}',question='Name a digit.',aliases=['2']) for i in range(2)]
            freq=torch.zeros(GA.text_config(E.model).vocab_size)
            token=E.encode('2',chat=False)
            with patch.object(E,'decode',side_effect=[[dict(ids=[],text='')],[dict(ids=token,text='2')]]):
                result,union=GK._collect_test(ctx,E,rows,{'H_op':[[0,1]]},'s3',freq)
            self.assertEqual(result['n_requested'],2);self.assertEqual(len(result['omitted']),1)
            self.assertEqual(result['ans'].shape,(1,1));self.assertEqual(result['omitted'][0]['row']['qid'],'q0')
            with patch.object(E,'decode',return_value=[dict(ids=[],text='')]):
                emptyrows=[dict(r,qid='empty_'+r['qid']) for r in rows]
                result,_=GK._collect_test(ctx,E,emptyrows,{'H_op':[[0,1]]},'s3',freq)
            self.assertIsNone(result['ans']);self.assertEqual(len(result['omitted']),2)

    def test_full_native_mock_prepare_measure_and_inherited_rules(self):
        with tempfile.TemporaryDirectory() as td, patch('socket.socket.connect',side_effect=AssertionError('network accessed')):
            model=GA.load_model('gemma4-e2b',mock=True)
            data=fixture_data()
            prep=MiniCtx(td,'k_prepare',data,model)
            status=GK.prepare(prep)
            self.assertEqual(status['status'],'done')
            hn=RC.load_json(prep.path('hn.json'))
            self.assertEqual(hn['layout']['widths'],[48,48,96,96])
            self.assertTrue(hn['H_op'])
            self.assertTrue(hn['mock'])
            ref=torch.load(prep.path('references.pt'),weights_only=False)
            self.assertEqual(ref['z_mu'].shape,(288,))
            self.assertEqual(len(ref['ref_r16']),2)
            for row in ref['ref_r16']:
                self.assertEqual(row['positions'][0],len(row['prompt_ids'])-1)
                self.assertEqual(len(row['positions']),len(row['continuation_ids']))
            measured=MiniCtx(td,'k_measure',data,model)
            GK.measure(measured)
            analyzed=MiniCtx(td,'k_analyze',data,model)
            result=GK.analyze(analyzed)
            self.assertEqual(result['status'],'done')
            verdict=RC.load_json(analyzed.path('k3_verdicts.json'))
            import r1b_k_rules as inherited
            d=torch.load(measured.path('k3_data.pt'),weights_only=False)
            self.assertEqual(verdict['K3a'],RC.jsonable(inherited.k3a_rule(d,quick=True)))
            self.assertIn('K3g_frame',verdict)
            self.assertEqual(verdict['other_stages']['K3d'],'k_causal')
            causal=MiniCtx(td,'k_causal',data,model)
            self.assertEqual(GK.causal(causal)['coverage'],['K3d'])
            cd=torch.load(causal.path('k3d_data.pt'),weights_only=False)
            self.assertEqual(set(cd['items']),{'ACH','TQA','NE','FP','MC','SY'})
            import r1b_k_ana as original_analysis
            cv=RC.load_json(causal.path('k3d_verdicts.json'))
            expected=original_analysis.k3d_stats(cd['rows'],cd['reference'])
            if cv['G_K9']:
                self.assertEqual(cv['verdict'],RC.jsonable(expected))
            # Explicit synthetic lens fixture tests the readout/verifier path;
            # it is not fitted or reported as scientific bridge evidence.
            lens=MiniCtx(td,'lens_fit',data,model)
            D=GA.text_config(model[1]).hidden_size
            lens.save_pt('lens.pt',{'maps':torch.eye(D).repeat(4,1,1),'mock':True})
            validation=MiniCtx(td,'lens_validate',data,model)
            validation.write_json('lens_validation.json',{'G_K_J':{'median_cos':{'output':1.}},'mock':True})
            readouts=MiniCtx(td,'k_readouts',data,model)
            import r1b_g_k_readouts as RO
            rs=RO.measure(readouts)
            self.assertIn(rs['status'],('done','partial'))
            rr=RC.load_json(readouts.path('k3e_readouts.json'))
            self.assertFalse(rr['external_catalogue_used'])
            for pole,row in rr['readouts'].items():
                l,j,p=pole.split(':');fid=lens._model[1]._r1b_g_layout.encode(int(l),int(j)) if hasattr(lens._model[1],'_r1b_g_layout') else GA.NeuronLayout.from_model(model[1]).encode(int(l),int(j))
                expected_thr=ref['p998_pos'][fid] if p=='+' else -ref['p998_neg'][fid]
                self.assertEqual(row['thr'],float(expected_thr))
            # Cached replay must not sample new model responses.
            with patch('r1b_g_engine.ModelEngine.decode',side_effect=AssertionError('uncached rerun')):
                GK.prepare(prep)


if __name__=='__main__':unittest.main(verbosity=2)
