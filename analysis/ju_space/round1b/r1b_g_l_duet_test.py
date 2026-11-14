"""Native Gemma L4 protocol checks with explicit synthetic populations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import r1b_core as K
from r1b_g_mock import make_mock
from r1b_g_adapter import NeuronLayout
from r1b_g_l_test import Context
from r1b_g_data import mock_data
from r1b_g_l_duet import run, sequence_choices
from r1b_l_rules import rule_L4

torch.set_num_threads(1)


class DuetTests(unittest.TestCase):
    def test_native_five_conditions_and_inherited_statistics(self):
        with tempfile.TemporaryDirectory() as td, patch('socket.socket.connect', side_effect=AssertionError('network')):
            pair = make_mock('e2b')
            tok, model = pair
            data = mock_data()
            cx = Context(td, 'l_duet', pair, data)
            cx.path = lambda rel: str(cx.work / rel)
            prep = Path(td) / 'l_prepare'; prep.mkdir()
            kprep = Path(td) / 'k_prepare'; kprep.mkdir()
            layout = NeuronLayout.from_model(model)
            pop = {'status':'measured', 'P':[0,48], 'CTRL':[[1,49],[2,50]], 'sg8_ok':False}
            (prep/'l_pop.json').write_text(json.dumps(pop))
            (kprep/'hn.json').write_text('{}')
            rows = [{'id':f'q{i}', 'split':'S_L', 'question':f'Which code is {i}?',
                     'ans0':tok(str(i),add_special_tokens=False)['input_ids'], 'aliases':[str(i)]} for i in range(8)]
            (prep/'items.json').write_text(json.dumps({'items':rows}))
            torch.save({'mu_z':torch.zeros(layout.total),'sd_z':torch.ones(layout.total),
                        'mu_cett':torch.zeros(layout.total),'sd_cett':torch.ones(layout.total)},prep/'refw.pt')
            ids=tok('A river flows.',add_special_tokens=False)['input_ids']
            torch.save({'ref_r16':[{'ids':ids,'positions':list(range(len(ids)-1))} for _ in range(2)]},kprep/'references.pt')
            result=run(cx)
            self.assertEqual(result['status'],'done')
            for cond in ('C-opp','C-half','C-doc','C-Hup','C-Rup'):
                out=torch.load(cx.work/f'l4_{cond}.pt',weights_only=False)
                self.assertEqual(len(out['runs']),16)
                self.assertEqual(len(out['runs'][0]['msgs']),2 if cond=='C-doc' else 4)
            stats=json.loads((cx.work/'l4.json').read_text())
            self.assertIn('J_ex',stats)
            self.assertIn('DD',stats)
            self.assertIn('dpsi',stats)
            self.assertEqual(stats['g_duet'], stats['auroc_first'] >= .8 and
                             min(stats['complete'].values()) >= .9)
            self.assertIn('pair', rule_L4(stats))
            tokenization=json.loads((cx.work/'l4_tokenization.json').read_text())
            self.assertTrue(tokenization['distinct'])
            self.assertFalse(tokenization['first_token_distinct'])
            self.assertFalse(any(b.mlp.down_proj._forward_hooks or b.mlp.down_proj._forward_pre_hooks
                                 for b in model.model.layers))

    def test_missing_raw_duet_input_is_explicit(self):
        from r1b_g_l_duet import d1
        with tempfile.TemporaryDirectory() as td:
            cx=Context(td,'l_duet',make_mock(),{})
            with self.assertRaisesRegex(K.NotRun,'raw F-sym'):
                d1(cx)

    def test_complete_value_probability_when_first_tokens_collide(self):
        tok, model=make_mock('12b')
        prefix=K.enc(tok,'The colour is')
        candidates=[K.enc(tok,' red'),K.enc(tok,' blue')]
        self.assertEqual(candidates[0][0],candidates[1][0])
        actual=sequence_choices(model,tok,[prefix],[candidates])[0]
        expected=[]
        for candidate in candidates:
            ids=torch.tensor([prefix+candidate])
            with torch.no_grad():
                lp=model(ids,use_cache=False).logits[0,len(prefix)-1:-1].float().log_softmax(-1)
            expected.append(float(lp.gather(1,torch.tensor(candidate)[:,None]).sum()))
        import numpy as np
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        self.assertNotEqual(actual[0],actual[1])


if __name__=='__main__':
    unittest.main(verbosity=2)
