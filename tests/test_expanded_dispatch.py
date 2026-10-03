import hashlib,json
from pathlib import Path
import pytest
from jev_observatory.expanded_dispatch import dispatch

def plan(tmp):
 p={'state':'α','model':'jev-1.13.0','questions':{'q0':{'type':'choice','instructions':'pick','criteria':{'o0':'a','o1':'b'}}}}
 b=json.dumps(p,ensure_ascii=False,separators=(',',':')).encode(); d={'schema':1,'calls':[{'request_id':'x','payload':p,'payload_sha256':hashlib.sha256(b).hexdigest()}]}; f=tmp/'p.json';f.write_text(json.dumps(d,ensure_ascii=False)); return f
class R:
 status_code=200; headers={}; content=b'{"model":"jev-1.13.0","usage":{"input_tokens":4},"answers":{}}'
class C:
 def post(self,*a,**k): return R()
def test_success_bytes_and_capture(tmp_path,monkeypatch):
 monkeypatch.setenv('JEVO_ALLOW_LIVE','1'); led=tmp_path/'l.json'; from jev_observatory.revision_run import PersistentBudget; PersistentBudget(led,20)
 out=tmp_path/'out'; s=dispatch(plan(tmp_path),out,led,key='secret-key',client=C()); assert s['attempted']==1
 assert json.loads((out/'attempts.jsonl').read_text())['status']==200
 assert 'secret-key' not in (out/'attempts.jsonl').read_text()
def test_bad_hash_zero_post(tmp_path,monkeypatch):
 monkeypatch.setenv('JEVO_ALLOW_LIVE','1'); f=plan(tmp_path); d=json.loads(f.read_text()); d['calls'][0]['payload_sha256']='bad'; f.write_text(json.dumps(d)); from jev_observatory.revision_run import PersistentBudget; led=tmp_path/'l.json'; PersistentBudget(led,20)
 with pytest.raises(ValueError): dispatch(f,tmp_path/'out',led,key='x',client=C())
 assert not (tmp_path/'out').exists()
