"""Model-free fake-Drive checks of K3 source selection, staging and failures."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

spec=importlib.util.spec_from_file_location('k3_builder',Path(__file__).with_name('build_runde1b_k3_cpu_notebook.py'))
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
FOLDER='application/vnd.google-apps.folder'


class Request:
    def __init__(self,value):self.value=value
    def execute(self):return self.value


class FakeDrive:
    def __init__(self):self.items={};self.raw={};self.media_reads=[]
    def files(self):return self
    def add(self,key,name,parent=None,raw=None,folder=False):
        self.items[key]=dict(id=key,name=name,mimeType=FOLDER if folder else 'application/octet-stream',
                             parents=[parent]if parent else [],size=str(len(raw))if raw else'0')
        if raw is not None:self.raw[key]=raw
        return key
    def get(self,fileId,**kw):return Request(self.items[fileId])
    def get_media(self,fileId,**kw):
        self.media_reads.append(fileId);return Request(self.raw[fileId])
    def list(self,q,**kw):
        name=re.search(r"name = '([^']+)'",q);parent=re.search(r"'([^']+)' in parents",q)
        rows=[v for v in self.items.values()if(not name or v['name']==name[1])and(not parent or parent[1]in v['parents'])and not v.get('trashed')]
        if "mimeType !="in q:rows=[v for v in rows if v['mimeType']!=FOLDER]
        return Request(dict(files=rows))


def fixture():
    service=FakeDrive()
    last=None
    for key,name in [('variants','variants'),('v2','htq_v2'),('session','r1b_K_S1'),('k','k'),('k3','k3')]:
        service.add(key,name,last,folder=True);last=key
    service.add('hn','hn','session',folder=True)
    payload=b'saved scientific tensor bytes; never unpickled by notebook helpers'
    digest=hashlib.sha256(payload).hexdigest()
    context={'hn/htq_policy.json':dict(h_policy='htq_v2',H_op_name='H_tq',post_hoc=True),
             'hn/hn_set.json':dict(h_policy='htq_v2',H_op_name='H_tq',post_hoc=True),
             'sg8_preview.json':dict(H_op_name='H_tq')}
    blobs={k:json.dumps(v).encode()for k,v in context.items()}
    done=dict(session='r1b_K_S1',mock=False,h_policy='htq_v2',files={B.REL:digest,
              **{k:hashlib.sha256(v).hexdigest()for k,v in blobs.items()}})
    blobs['session_done.json']=json.dumps(done).encode()
    for rel,raw in blobs.items():
        service.add(rel,rel.split('/')[-1],'hn'if rel.startswith('hn/')else'session',raw)
    service.add('data','k3a_data.pt','k3',payload)
    return service,payload,blobs


class SourceTests(unittest.TestCase):
    def setUp(self):self.api,self.payload,self.blobs=fixture()
    def test_auto_finds_only_canonical_complete_source(self):
        self.api.add('other','k3a_data.pt',None,b'wrong')
        rows=B.api_candidates(self.api)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['file_id'],'data')
        self.assertNotIn('data',self.api.media_reads) # No big download during discovery.
        self.assertEqual(B.choose_source(rows)['sha256'],hashlib.sha256(self.payload).hexdigest())
    def test_wrong_canonical_parent_and_mock_fail_closed(self):
        self.api.items['v2']['name']='wrong_variant'
        with self.assertRaisesRegex(ValueError,'Quellpfad'):B.api_context(self.api,'data')
        done=json.loads(self.blobs['session_done.json']);done['mock']=True
        self.blobs['session_done.json']=json.dumps(done).encode()
        with self.assertRaisesRegex(ValueError,'echter'):B.checked_context(self.blobs)
    def test_conflicting_copy_requires_explicit_choice(self):
        a=B.api_context(self.api,'data');b=dict(a,sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'DATEI_ID'):B.choose_source([a,b])
        self.assertIs(B.choose_source([a,dict(a)]),a)
    def test_context_tamper_rejected_but_extra_partial_does_not_override_valid_artifact_hash(self):
        self.api.raw['hn/hn_set.json']=b'{}'
        with self.assertRaisesRegex(ValueError,'Kontext-Prüfsumme'):B.api_context(self.api,'data')
        self.api,self.payload,self.blobs=fixture()
        self.api.add('partial','session_partial.json','session',b'{}')
        self.assertTrue(B.api_context(self.api,'data')['source_partial_present'])
    def test_streaming_stage_and_sha_cache_resume(self):
        source=B.api_context(self.api,'data')
        calls=[]
        class Download:
            def __init__(self,dst,req,chunksize):self.dst,self.req=dst,req;calls.append(chunksize)
            def next_chunk(self,num_retries):
                self.dst.write(self.req.execute());return SimpleNamespace(progress=lambda:1.),True
        with tempfile.TemporaryDirectory()as tmp:
            session=B.stage_source(source,tmp,self.api,Download)
            self.assertEqual((session/B.REL).read_bytes(),self.payload)
            self.assertEqual(calls,[16<<20])
            for rel in B.CONTEXT:self.assertEqual((session/rel).read_bytes(),self.blobs[rel])
            B.stage_source(source,tmp,self.api,lambda *a,**k: self.fail('valid cache redownloaded'))
    def test_bad_download_never_becomes_loadable_target(self):
        source=B.api_context(self.api,'data');source['size']=3
        class Download:
            def __init__(self,dst,req,**kw):self.dst=dst
            def next_chunk(self,**kw):self.dst.write(b'bad');return None,True
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaisesRegex(ValueError,'SHA256'):B.stage_source(source,tmp,self.api,Download)
            self.assertFalse(list(Path(tmp).rglob('k3a_data.pt')))
            self.assertTrue(list(Path(tmp).rglob('*.download')))
    def test_existing_mounted_paths_preferred_and_source_untouched(self):
        with tempfile.TemporaryDirectory()as tmp:
            s=Path(tmp)/'dsse_runs_B/variants/htq_v2/r1b_K_S1'
            for rel,raw in {B.REL:self.payload,**self.blobs}.items():
                p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            sources=B.mounted_candidates(tmp)
            self.assertEqual(len(sources),1)
            local=B.stage_source(sources[0],Path(tmp)/'cache')
            self.assertEqual((s/B.REL).read_bytes(),self.payload)
            self.assertEqual((local/B.REL).read_bytes(),self.payload)


class NotebookTests(unittest.TestCase):
    def test_generated_metadata_syntax_and_no_model_or_mount_commands(self):
        with tempfile.TemporaryDirectory()as tmp:n=json.loads(B.build(tmp).read_text())
        self.assertNotIn('accelerator',n['metadata']);self.assertNotIn('gpuType',n['metadata']['colab'])
        for cell in n['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        s='\n'.join(''.join(c['source'])for c in n['cells']if c['cell_type']=='code')
        for forbidden in ['drive.mount(', 'fetch-models', 'from_pretrained(', 'r1b_run_session.py','runtime.unassign(']:self.assertNotIn(forbidden,s)
        self.assertIn('DATEI_ID = ""',s)
        self.assertIn('r1b_k3_cpu_export.py',s)
        self.assertNotIn('719671301',s) # No private input-specific constants.
    def test_gpu_and_auth_failures_produce_downloadable_diagnostic(self):
        for failure in ['gpu','auth']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory()as tmp:
                google=ModuleType('google');colab=ModuleType('google.colab')
                colab.files=SimpleNamespace(download=Mock());colab.auth=SimpleNamespace(authenticate_user=Mock(side_effect=RuntimeError('credential propagation was unsuccessful')))
                google.colab=colab
                packages={name:SimpleNamespace(__version__='test')for name in ['numpy','scipy','torch']}
                packages['torch'].cuda=SimpleNamespace(is_available=lambda:False)
                scope=dict(BRANCH=B.BRANCH,DATEI_ID='',AUSGABE_ROOT=tmp,CACHE_ROOT=tmp+'/cache',REL=B.REL,
                           mounted_candidates=lambda p:[],sha256=B.sha256)
                result=SimpleNamespace(returncode=0 if failure=='gpu' else 1,stdout='A100'if failure=='gpu'else'')
                with patch.dict(sys.modules,{'google':google,'google.colab':colab,**packages}),patch.dict(os.environ,{'COLAB_GPU':'0'}),patch('subprocess.run',return_value=result),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'K3-CPU wurde angehalten'):
                        exec(compile(B.main_source(),'K3 CPU','exec'),scope)
                archive=next(Path(tmp).rglob('htq_k3_diagnostic.zip'))
                with zipfile.ZipFile(archive)as z:
                    status=json.loads(z.read('notebook_status.json'))
                    self.assertEqual(status['state'],'failed')
                    self.assertEqual(status['stage'],'cpu_check'if failure=='gpu'else'source_context')
                    self.assertIn('notebook.log',z.namelist())
                colab.files.download.assert_called_once()
                self.assertEqual(colab.auth.authenticate_user.call_count,0 if failure=='gpu' else 1)


if __name__=='__main__':unittest.main()
