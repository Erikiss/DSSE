"""Stored-data K3 CPU export: provenance, exact projection and OOF isolation.

Run with the CPU test environment:
    python -m unittest discover -s analysis/ju_space/round1b/tests -p test_k3_cpu_export.py
The fixture contains no pretrained weights, tokenizer or natural-language inputs.
"""
import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

R1B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1B))
import r1b_k3_cpu_export as EXPORT
import r1b_k_rules as KR
import r1b_k_base as KB
import r1b_core as RC

SOURCE_COMMIT = 'de28147e5f0cd8932947ae8254e045dd4388f560'


def fixture(with_random=False):
    """Unused columns and reversed set order expose silent projection mistakes."""
    rng = np.random.default_rng(9173)
    union = [(0, 4), (0, 12), (2, 5), (2, 31), (3, 16), (6, 2)]
    hop, har = [union[4], union[1]], [union[2]]
    sets = {'H_op': hop, 'H_tq': hop, 'H_ar': har, 'H+': union}
    if with_random:
        sets['RS0'] = [union[5]]
    yq = np.tile([0, 1], 40).astype(np.int64)
    pre = rng.normal(size=(80, 6)).astype(np.float32)
    pre[:, 1] += .4 * yq
    pre[:, 4] += .7 * yq
    ans = rng.normal(size=(80, 6)).astype(np.float32)
    ans[:, 1] += 1.2 * yq
    ans[:, 4] += .9 * yq
    other = rng.normal(size=(80, 6)).astype(np.float32)
    fit = dict(X=np.r_[ans, other], y=np.r_[yq, np.zeros(80, dtype=np.int64)], pre=pre, y_q=yq)

    def family(labels):
        n = len(labels)
        y = np.array([x in ('wrong', 'answered') for x in labels], dtype=float)
        a, p = rng.normal(size=(n, 6)), rng.normal(size=(n, 6))
        a[:, 1] += 1.1*y; a[:, 4] += .8*y
        p[:, 1] += .5*y; p[:, 4] += .4*y
        cov = [dict(ent=float(rng.uniform(.3, 2)), lp=float(rng.uniform(-4, -.2)),
                    len=5+i%11, freq=float(rng.uniform(-10, -2)), pre_ent=float(rng.uniform(.3, 3))) for i in range(n)]
        return dict(y=labels, ans=torch.tensor(a, dtype=torch.float16), pre=torch.tensor(p, dtype=torch.float16),
                    cov=cov, conts=[[1, 2, 3] for _ in range(n)])
    return dict(union=union, sets=sets, fit=fit,
                har=dict(X=rng.normal(size=(80, 6)).astype(np.float32), y=np.tile([0, 1], 40)),
                s3=family(['wrong']*48+['correct']*32+['near_miss']*2+['refusal']),
                adet=family(['wrong']*12+['correct']*12),
                ne=family(['answered']*12+['refusal']*12),
                qids_s3=[f'private-fixture-{i}' for i in range(83)])


def statistics(data):
    """Call the unchanged production numerical rules, including B=2000."""
    return dict(K3a=KR.k3a_rule(data, quick=False), K3b=KR.k3b_rule(data), K3c=KR.k3c_rule(data))


def assert_nested_close(case, left, right):
    if isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            assert_nested_close(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_close(case, a, b)
    elif isinstance(left, (int, float, np.number)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        case.assertEqual(left, right)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(RC.jsonable(obj), allow_nan=False))


def session_fixture(root, data=None):
    """A coherent producer manifest, not an allowlist bypass of the validator."""
    data = fixture() if data is None else data
    session = Path(root)/'r1b_K_S1'
    source = session/EXPORT.DATA_REL
    source.parent.mkdir(parents=True)
    torch.save(data, source)
    provenance = dict(session='r1b_K_S1', step='k3a', h_policy='htq_v2', post_hoc=True,
                      amendment=2, code_commit=SOURCE_COMMIT, mock=False, inputs=[],
                      modules={'analysis/ju_space/round1b/'+k:v for k,v in EXPORT.CODE_HASHES.items()})
    write_json(source.with_name(source.name+'.prov.json'),
               dict(file=source.name, sha256=digest(source), provenance=provenance))
    hs = dict(h_policy='htq_v2', post_hoc=True, amendment=2, H_op_name='H_tq',
              H_op=data['sets']['H_op'], H_tq=data['sets']['H_tq'], H_ar=data['sets']['H_ar'],
              H_plus=data['sets']['H+'], F=[], n_H=len(data['sets']['H_op']), provenance=provenance)
    write_json(session/'hn/hn_set.json', hs)
    write_json(session/'hn/htq_policy.json', dict(h_policy='htq_v2', post_hoc=True,
               amendment=2, H_op_name='H_tq', n_H_op=len(data['sets']['H_op']), provenance=provenance))
    preview = KR.k3a_rule(data, quick=True)
    write_json(session/'sg8_preview.json', dict(detail=preview, provenance=provenance))
    rels = [EXPORT.DATA_REL, EXPORT.DATA_REL+'.prov.json', *EXPORT.CONTEXT_RELS]
    hashes = {rel:digest(session/rel) for rel in rels}
    write_json(session/'session_done.json', dict(session='r1b_K_S1', mock=False,
        h_policy='htq_v2', code_commit=SOURCE_COMMIT, files=hashes, not_run={}, finished='2026-11-17T00:00:00Z',
        steps={'k3a':dict(state='ran', outputs={rel:dict(sha256=hashes[rel],size=(session/rel).stat().st_size)
                                             for rel in [EXPORT.DATA_REL, EXPORT.DATA_REL+'.prov.json']})}))
    return source, session


def update_source_hash(session, rel):
    path = session/'session_done.json'; done=json.loads(path.read_text())
    done['files'][rel]=digest(session/rel)
    if rel in done['steps']['k3a']['outputs']:
        done['steps']['k3a']['outputs'][rel]={'sha256':done['files'][rel], 'size':(session/rel).stat().st_size}
    write_json(path,done)


def snapshot(path):
    return {str(p.relative_to(path)):digest(p) for p in path.rglob('*') if p.is_file()}


def pairwise_auc(y, scores):
    """Independent probability-of-ranking definition, including half credit for ties."""
    positive=scores[y==1][:,None]
    negative=scores[y==0][None,:]
    return float(np.mean((positive>negative)+.5*(positive==negative)))


class CompactTests(unittest.TestCase):
    def test_compact_preserves_all_three_frozen_rules_and_does_not_mutate_original(self):
        for with_random in (False, True):
            with self.subTest(random=with_random), tempfile.TemporaryDirectory() as td:
                data = fixture(with_random)
                before = copy.deepcopy(data)
                path = Path(td)/'compact.npz'
                EXPORT.make_compact(data, path)
                compact = EXPORT.load_compact(path)
                needed = set(data['sets']['H_op']+data['sets']['H_ar']+data['sets'].get('RS0', []))
                self.assertEqual(list(map(tuple, compact['union'])), [n for n in data['union'] if n in needed])
                self.assertEqual(list(map(tuple, compact['sets']['H_op'])), data['sets']['H_op'])
                self.assertEqual(list(map(tuple, compact['sets']['H_ar'])), data['sets']['H_ar'])
                self.assertEqual(compact['fit']['X'].shape[1], len(needed))
                self.assertLess(compact['fit']['X'].nbytes, data['fit']['X'].nbytes)
                assert_nested_close(self, statistics(data), statistics(compact))
                self.assertEqual(data['union'], before['union'])
                self.assertEqual(data['sets'], before['sets'])
                np.testing.assert_array_equal(data['fit']['X'], before['fit']['X'])
                torch.testing.assert_close(data['s3']['ans'], before['s3']['ans'], rtol=0, atol=0)

    def test_heldout_fold_scores_do_not_depend_on_its_own_labels(self):
        data = fixture()
        cols = KR._cols(data['union'], data['sets']['H_op'])
        y = KR._ybin(data['s3']['y']); keep=np.flatnonzero(y>=0)
        folds=np.arange(len(keep))%5
        x=data['s3']['pre'].numpy()[keep][:,cols]
        original=KB.crossfit_auroc(x,y[keep],folds)
        changed=y[keep].copy(); changed[folds==0]=1-changed[folds==0]
        rescored=KB.crossfit_auroc(x,changed,folds)
        np.testing.assert_allclose(original[folds==0],rescored[folds==0],rtol=0,atol=0)
        self.assertGreater(np.max(np.abs(original[folds!=0]-rescored[folds!=0])),1e-6)

    def test_single_class_arithmetic_and_tiny_ne_preserve_expected_undefined_codes(self):
        data=fixture()
        data['adet']['y']=['wrong']*len(data['adet']['y'])
        take=[0,1,12,13]
        for key in ['y','cov','conts']:data['ne'][key]=[data['ne'][key][i] for i in take]
        for key in ['pre','ans']:data['ne'][key]=data['ne'][key][take]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'compact.npz';EXPORT.make_compact(data,path)
            before=KR.k3a_rule(data,quick=False)
            after=KR.k3a_rule(EXPORT.load_compact(path),quick=False)
        assert_nested_close(self,before,after)
        self.assertEqual(before['arith']['code']['code'],'NaN:no_test_rows_or_set')
        self.assertEqual(before['har']['code']['code'],'NaN:no_test_rows_or_set')
        self.assertEqual(before['ne']['n'],4)
        self.assertEqual(before['ne']['auroc_ci']['lo']['code'],'NaN:fewer_than_95pct_finite_replicates')
        self.assertEqual(before['ne']['string'],'undecided')


class SourceValidationTests(unittest.TestCase):
    def test_valid_manifest_records_all_verified_input_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            source,session=session_fixture(td)
            result=EXPORT.validate_source(source,session)
            self.assertEqual(result['files'][EXPORT.DATA_REL]['sha256'],digest(source))
            self.assertEqual(result['manifest_sha256'],digest(session/'session_done.json'))
            self.assertEqual(result['done']['code_commit'],SOURCE_COMMIT)

    def test_tampered_tensor_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as td:
            source,session=session_fixture(td)
            with source.open('ab') as f:f.write(b'changed')
            with patch.object(KB,'load_pt',side_effect=AssertionError('unverified pickle loaded')) as loader:
                with self.assertRaisesRegex(ValueError,'SHA256'):
                    EXPORT.run_export(source,session,Path(td)/'exports')
            loader.assert_not_called()
            self.assertFalse(list((Path(td)/'exports').rglob('results.json')))

    def test_wrong_policy_and_source_commit_are_rejected(self):
        for field,value in [('h_policy','original'),('code_commit','0'*40),('mock',True)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as td:
                source,session=session_fixture(td)
                path=session/'session_done.json'; done=json.loads(path.read_text());done[field]=value;write_json(path,done)
                with self.assertRaises(ValueError):EXPORT.validate_source(source,session)

    def test_sidecar_cannot_contradict_a_coherent_completion_hash(self):
        for field,value in [('sha256','0'*64),('file','some_other_file.pt')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as td:
                source,session=session_fixture(td); rel=EXPORT.DATA_REL+'.prov.json'
                path=session/rel; sidecar=json.loads(path.read_text());sidecar[field]=value;write_json(path,sidecar)
                update_source_hash(session,rel)
                with self.assertRaises(ValueError):EXPORT.validate_source(source,session)

    def test_semantic_bad_arrays_fail_before_numerical_results_are_written(self):
        cases={
            'duplicate_qid':lambda d:d['qids_s3'].__setitem__(1,d['qids_s3'][0]),
            'nonfinite_selected':lambda d:d['fit']['X'].__setitem__((0,4),float('nan')),
            'missing_covariate_row':lambda d:d['s3']['cov'].pop(),
            'missing_continuation_row':lambda d:d['s3']['conts'].pop(),
            'duplicate_union':lambda d:d['union'].__setitem__(0,d['union'][1]),
        }
        for name,mutate in cases.items():
            with self.subTest(case=name):
                data=fixture();mutate(data)
                with self.assertRaises(ValueError):
                    EXPORT.audit_data(data,dict(H_op=data['sets']['H_op']))


class ExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.data=fixture()
        cls.source,cls.session=session_fixture(cls.tmp.name,cls.data)
        cls.before=snapshot(cls.session)
        # These are model/download entry points, not statistical algorithms.
        with patch.object(RC,'load_model',side_effect=AssertionError('model inference is forbidden')), \
             patch.object(RC,'ensure_model',side_effect=AssertionError('model download is forbidden')):
            cls.output=EXPORT.run_export(cls.source,cls.session,Path(cls.tmp.name)/'exports')
        cls.run_dir=Path(cls.output['run_dir'])
        cls.results=json.loads((cls.run_dir/'results.json').read_text())
        with (cls.run_dir/'scores.csv').open() as f:cls.scores=list(csv.DictReader(f))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_is_exploratory_and_keeps_missing_controls_missing(self):
        self.assertFalse(self.results['confirmatory_eligible'])
        self.assertIn('post hoc',self.results['inference'])
        self.assertEqual(self.results['bootstrap_replicates'],2000)
        self.assertEqual(self.results['confidence_level'],.90)
        self.assertEqual(self.results['random_set_count'],0)
        self.assertEqual(self.results['K3a']['in_domain']['string'],'undecided')
        self.assertEqual(self.results['K3a']['in_domain']['auroc_RS'],[])
        self.assertFalse((self.run_dir/'k_verdict.json').exists())

    def test_complete_export_does_not_change_any_original_session_byte(self):
        self.assertEqual(snapshot(self.session),self.before)

    def test_csv_reconstructs_frozen_point_estimates_and_balancing(self):
        y=np.array([int(r['wrong']) for r in self.scores])
        score=lambda name:np.array([float(r[name]) for r in self.scores])
        balanced=np.array([int(r['in_k3a_balanced']) for r in self.scores],bool)
        self.assertEqual(len(y),80)
        self.assertEqual(int(balanced.sum()),64)
        self.assertEqual(np.bincount(y[balanced]).tolist(),[32,32])
        self.assertAlmostEqual(pairwise_auc(y[balanced],score('answer_fit')[balanced]),self.results['K3a']['in_domain']['auroc_H'],places=12)
        self.assertAlmostEqual(pairwise_auc(y,score('pre_fit')),self.results['K3b']['auroc_pre'],places=12)
        self.assertAlmostEqual(pairwise_auc(y,score('pre_answer_oof'))-pairwise_auc(y,score('pre_oof')),self.results['K3b']['delta'],places=12)
        self.assertAlmostEqual(pairwise_auc(y,score('covariates_H_oof'))-pairwise_auc(y,score('covariates_oof')),self.results['K3c']['delta'],places=12)

    def test_csv_oof_scores_ignore_heldout_labels(self):
        changed=copy.deepcopy(self.data)
        keep=np.flatnonzero(KR._ybin(changed['s3']['y'])>=0)
        for j,i in enumerate(keep):
            if j%5==0:
                changed['s3']['y'][i]='correct' if changed['s3']['y'][i]=='wrong' else 'wrong'
        path=Path(self.tmp.name)/'heldout_labels_changed.csv'
        EXPORT.score_export(changed,path)
        with path.open() as f:rows=list(csv.DictReader(f))
        for original,updated in zip(self.scores,rows):
            self.assertEqual(original['qid'],updated['qid'])
            if int(original['fold'])==0:
                for col in ['pre_oof','pre_answer_oof','covariates_oof','covariates_H_oof']:
                    self.assertEqual(float(original[col]),float(updated[col]))

    def test_output_under_original_session_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'außerhalb'):
            EXPORT.run_export(self.source,self.session,self.session/'new_exports')
        self.assertEqual(snapshot(self.session),self.before)


if __name__ == '__main__':
    unittest.main()
