"""Native tiny-Gemma checks for G chain, attribution and causal route semantics."""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import r1b_g_k_chain as K
from r1b_g_adapter import load_model
from r1b_g_engine import ModelEngine
from r1b_g_lens import frame_spec,rollout


class MemoryCtx:
    mock=True
    model_name="gemma4-e2b"
    freeze={}
    def __init__(self,pair):self.pair=pair;self.saved={};self.inputs={}
    def model(self):return self.pair
    def cached(self,name,fn):
        if name not in self.saved:self.saved[name]=fn()
        return self.saved[name]
    def key(self,*parts):
        import hashlib
        return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:15],16)
    def data(self):return self.inputs


class NativeChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pair=load_model("gemma4-e2b",mock=True,device="cpu")
    def setUp(self):
        self.ctx=MemoryCtx(self.pair);self.E=ModelEngine(self.ctx)
        self.seq=self.E.encode("A plain question about a quiet river and a bridge.")
    def test_foreign_controls_have_no_identity(self):
        ids=list("abcde");items={i:dict(template="same" if i!="e" else "singleton",nops=2,mul=False,digits=[2,2]) for i in ids}
        a,matched=K.foreign_indices(ids,items)
        self.assertTrue(all(i!=j for i,j in enumerate(a)))
        self.assertEqual(matched,[True]*4+[False])
        self.assertEqual(K.foreign_indices(ids,items)[0],a)
    def test_missing_operands_are_not_guessed(self):
        with self.assertRaisesRegex(ValueError,"operands"):
            K.normalize_item(dict(id="x",prompt="11+23",answer=34,intermediates=[]))
    def test_exact_parser_preserves_precedence(self):
        item=dict(id="x",prompt="Compute.",P=[11,23],I=[34],A=57,Iplus=[42])
        tokens=self.E.encode(" 34 57 11 42 88 ",chat=False)
        score=K.score_tokens(self.E.tok,tokens,item,OWN=[88])
        labels={value:kind for _,value,kind in score["claims"]}
        self.assertEqual(labels["34"],"TRUE-INT")
        self.assertEqual(labels["57"],"ANSWER")
        self.assertEqual(labels["11"],"PROMPT")
        self.assertEqual(labels["88"],"OWN")
    def test_frozen_qk_matches_native_across_sliding_boundary(self):
        seq=(self.seq*3)[:96]
        native=self.E.next_logits(seq)
        previous=self.E.cfg._attn_implementation
        with K.FrozenQK(self.E) as recorded:
            explicit=self.E.next_logits(seq)
        with K.FrozenQK(self.E,recorded.probabilities):
            replay=self.E.next_logits(seq)
        self.assertTrue(torch.allclose(native,explicit,atol=1e-6,rtol=1e-6))
        self.assertTrue(torch.equal(explicit,replay))
        self.assertEqual(self.E.cfg._attn_implementation,previous)
        from transformers import AttentionInterface
        self.assertNotIn("r1b_g_frozen_qk",AttentionInterface._global_mapping)
    def test_frozen_attention_restores_after_exception(self):
        previous=self.E.cfg._attn_implementation
        with self.assertRaisesRegex(RuntimeError,"fixture"):
            with K.FrozenQK(self.E):raise RuntimeError("fixture")
        self.assertEqual(self.E.cfg._attn_implementation,previous)
    def test_fd_matches_native_vjp_and_late_structural_zero(self):
        dimension=int(self.E.cfg.hidden_size)
        g=torch.Generator().manual_seed(841)
        v=torch.randn(dimension,generator=g);u=torch.randn(dimension,generator=g)
        last=len(self.E.layers)-1
        result=K.fd_vjp_check(self.E,self.seq,0,len(self.seq)-5,last,len(self.seq)-2,v,u,eps=.01)
        self.assertLess(result["relative_error"],.02)
        zero=K.fd_transfer(self.E,self.seq,last,3,v*.01,[last])[last][4:]
        self.assertEqual(float(zero.abs().max()),0.)
    def test_native_signed_shares_are_actual_bounded_and_finite(self):
        h=self.E.hidden([self.seq],[0])[0][0][-1]
        populations={"all":[self.E.layout.decode(i) for i in range(self.E.layout.total)],"H":[(0,0),(1,1)]}
        with K.NativeShares(self.E,populations) as shares:
            out=rollout(self.E,0,h,[4,5],steps=3,prior_state=torch.zeros_like(h),frame_ids=frame_spec(self.E)["frame_ids"],observer=shares.observe)
        self.assertTrue(bool(out["e"].abs().max()>0))
        for seed in range(2):
            for values in shares.row(seed).values():
                self.assertEqual(len(values),3)
                self.assertTrue(bool(torch.isfinite(values).all()))
                self.assertLessEqual(float(values.abs().max()),1.00001)
        self.assertGreater(float(shares.row(0)["all"].abs().max()),0.)
    def test_native_direct_writes_include_post_ff_normalization(self):
        captured={};block=self.E.layers[0]
        h=block.post_feedforward_layernorm.register_forward_hook(lambda m,a,o:captured.update(value=o[0,-1].detach().float()))
        try:
            populations={"all":[(0,j) for j in range(self.E.layout.widths[0])]}
            refs={"z_mu":torch.zeros(self.E.layout.total)}
            writes=K.direct_writes(self.E,self.seq,populations,refs,[0])
        finally:h.remove()
        expected=captured["value"]*float(torch.as_tensor(block.layer_scalar))
        self.assertTrue(torch.allclose(writes[0]["all"],expected,atol=1e-5,rtol=1e-5))
    def test_absolute_position_noise_matches_restarted_continuation(self):
        initial=self.seq[:8]
        whole=K._route_generation(self.ctx,self.E,initial,"fixture",8)
        restarted=K._route_generation(self.ctx,self.E,whole[:12],"fixture",4)
        self.assertEqual(whole,restarted)
    def test_read_state_interventions_execute_with_explicit_fixture_items(self):
        # Fixture eligibility is an explicit function argument, not a claim that
        # the random model solves these programs. No scientific artifact is made.
        items={"a":dict(id="a",prompt="Compute 11 + 23 + 20.",P=[11,23,20],I=[34],A=54,Iplus=[]),
               "b":dict(id="b",prompt="Compute 17 + 21 + 30.",P=[17,21,30],I=[38],A=68,Iplus=[])}
        blocks=K.mapped_blocks(len(self.E.layers));prompts=[self.E.encode(x["prompt"]) for x in items.values()]
        hidden=self.E.hidden(prompts,blocks)
        source=dict(ids=list(items),prompts=prompts,h={b:torch.stack([r[b][-1] for r in hidden]) for b in blocks})
        hn={"H_op":[(0,0)],"RS39mc":[[(0,1)],[(0,2)]]}
        refs={"z_mu":torch.zeros(self.E.layout.total)}
        gates={i:{"own":[],"W":[]} for i in items}
        result=K.causal_chain_controls(self.ctx,self.E,source,frame_spec(self.E)["frame_ids"],None,hn,refs,items,gates,list(items),blocks[1])
        self.assertEqual(result["status"],"measured")
        conditions={r["cond"] for r in result["k6c"]}
        self.assertTrue({"none","clamp_H","ablate_H","clamp_MC0","ablate_MC0"}<=conditions)
        self.assertEqual({r["variant"] for r in result["k6g"]},{"correct","wrong","neutral"})
        for row in result["k6c"]:
            self.assertTrue(all(torch.isfinite(torch.tensor(row["css"]))))


if __name__=="__main__":unittest.main()
