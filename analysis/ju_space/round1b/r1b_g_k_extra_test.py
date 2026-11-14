"""Offline tests of G K6f/h and K7a statistical/native model instruments."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from r1b_g_engine_test import Context
from r1b_g_engine import ModelEngine
from r1b_g_k_extra import (_probe,_phase_surrogate,_interventional_arrows,
                           context_offsets,qa_lens,atlas)
from r1b_g_data import mock_data


class InstrumentContext(Context):
    def __init__(self):
        super().__init__()
        self.temp=tempfile.TemporaryDirectory(prefix="g-extra-test-")
        self.root=Path(self.temp.name);self.raw=mock_data();self.cache={}
    def data(self):return self.raw
    def artifact(self,stage,relative):return self.root/stage/relative
    def cached(self,name,fn):
        if name not in self.cache:self.cache[name]=fn()
        return self.cache[name]


class ProbeTests(unittest.TestCase):
    def test_nested_text_blocked_ridge_recovers_forward_map(self):
        rng=np.random.default_rng(4)
        X=rng.normal(size=(100,4));W=rng.normal(size=(4,6))
        Y=(X@W).reshape(100,2,3);groups=np.repeat(np.arange(10),10)
        result=_probe(X,Y,groups,lambdas=(1e-6,.1,100.))
        self.assertTrue(min(result["r2"])>.999)
        self.assertEqual(len(result["per_text"]),10)
        self.assertEqual(len(result["lambdas"]),5)
        self.assertTrue(all(r["lambda"]==[1e-6,1e-6] for r in result["lambdas"]))

    def test_common_phase_preserves_multivariate_spectra(self):
        rng=np.random.default_rng(9);X=rng.normal(size=(64,3));groups=np.repeat([0,1],32)
        Y=_phase_surrogate(X,groups,np.random.default_rng(12))
        self.assertFalse(np.allclose(X,Y))
        for group in (0,1):
            a=np.fft.rfft(X[groups==group],axis=0);b=np.fft.rfft(Y[groups==group],axis=0)
            np.testing.assert_allclose(a[:,0]*a[:,1].conj(),b[:,0]*b[:,1].conj(),atol=1e-10)

    def test_interventional_arrow_uses_matched_random_gains(self):
        data={"channels":{"a":0,"c":1,"b":2,"rand_c_0":1},"random_for":{"c":["rand_c_0"]},
              "rows":[{"g":{"a->c":2.,"c->b":8.,"a->rand_c_0":1.,"rand_c_0->b":2.}}]}
        result=_interventional_arrows(data)
        self.assertAlmostEqual(result["c"]["alpha_int"],np.log(2))
        self.assertEqual(result["a"]["status"],"undefined")


class NativeExtraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ctx=InstrumentContext();cls.E=ModelEngine(cls.ctx)
    @classmethod
    def tearDownClass(cls):cls.ctx.temp.cleanup()

    def test_context_offsets_cached_full_parity_and_pairing(self):
        E,ctx=self.E,self.ctx
        items=ctx.raw["arithmetic"][:4];prompts=[E.encode(it["prompt"]) for it in items]
        states=torch.stack([r[0][-1] for r in E.hidden(prompts,[0])]);ids=[it["id"] for it in items]
        source={"source":{"t0":{"ids":ids,"prompts":prompts,"h":{0:states}}},
                "correct":ids[:2],"wrong":ids[2:],"items":{it["id"]:it for it in items}}
        chains={"verifier":{"b_star":0},"gating":{"gate":{i:{"own":[],"W":[]} for i in ids}}}
        out=context_offsets(ctx,E,source,chains,{},None,None)
        self.assertEqual(out["status"],"measured")
        self.assertEqual(len(out["rows"]),16)
        self.assertTrue(all(x["ok"] for x in out["parity"].values()))
        self.assertEqual({r["variant"] for r in out["rows"]},{"V1_F16","V1_F32"})
        self.assertTrue(all(torch.isfinite(r["lpr"]).all() for r in out["rows"]))

    def test_qa_foreign_controls_never_cross_class(self):
        ctx=self.ctx
        rows=[]
        for cls in ("CC","CI"):
            for j in range(2):
                rows.append({"qid":f"{cls}-{j}","question":f"Who wrote book {j}?","aliases":["Ada"],
                             "split":"S_lens","cls":cls,"answers":[{"text":"Ada"},{"text":"Ada"},{"text":"Bea"}]})
        p=ctx.artifact("k_prepare","lab.json");p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({"items":rows}))
        out=qa_lens(ctx,self.E,{},None,None)
        self.assertEqual(out["status"],"measured")
        self.assertEqual(len(out["rows"]),16)
        for ident,control in out["controls"].items():
            self.assertNotEqual(ident,control["foreign"])
            self.assertEqual(ident[:2],control["foreign"][:2])
            self.assertEqual(control["modal"],"Ada")
        self.assertFalse(out["G_K_J"])

    def test_atlas_covers_every_native_head_and_marks_exploratory(self):
        out=atlas(self.ctx,self.E,{},None,None)
        expected=sum(layer.self_attn.o_proj.in_features//layer.self_attn.head_dim for layer in self.E.layers)
        self.assertEqual(out["head_count"],expected)
        self.assertEqual(out["status"],"measured")
        self.assertIn("never counted",out["note"])
        self.assertTrue(out["ceilings"])
        self.assertTrue(out["dynamics"]["D"])


if __name__=="__main__":unittest.main()
