"""Fail-closed runner for immutable native frozen plans."""
from __future__ import annotations
import copy, hashlib, json, math, os, time, uuid
from pathlib import Path
from typing import Any
from .revision_run import PersistentBudget
from .redact import Redactor

MAX_INPUT = 65536
RATE = .042 / 1_000_000

def _bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

def validate_plan(plan: dict) -> list[tuple[str, bytes, dict]]:
    if plan.get("schema") != 1 or not isinstance(plan.get("calls"), list): raise ValueError("invalid frozen plan")
    result=[]; ids=set()
    for c in plan["calls"]:
        rid=c.get("request_id")
        if not isinstance(rid,str) or rid in ids: raise ValueError("duplicate request id")
        ids.add(rid); p=copy.deepcopy(c.get("payload")); b=_bytes(p)
        if c.get("payload_sha256") != hashlib.sha256(b).hexdigest(): raise ValueError("payload hash mismatch")
        if set(p) != {"state","model","questions"} or not isinstance(p["state"],str) or p["model"]!="jev-1.13.0": raise ValueError("native payload")
        if set(p["questions"]) != {"q0"}: raise ValueError("q0 required")
        q=p["questions"]["q0"]
        if set(q) != {"type", "instructions", "criteria"} or q.get("type")!="choice" or not isinstance(q.get("instructions"),str): raise ValueError("choice contract")
        cr=q.get("criteria")
        if not isinstance(cr,dict) or not 2<=len(cr)<=255 or list(cr)!=[f"o{i}" for i in range(len(cr))] or not all(isinstance(k,str) and isinstance(v,str) for k,v in cr.items()): raise ValueError("criteria contract")
        result.append((rid,b,p))
    return result

def dispatch(plan_path, output, ledger_path, cap=20, key=None, concurrency=1, client=None):
    if os.getenv("JEVO_ALLOW_LIVE") != "1": raise RuntimeError("JEVO_ALLOW_LIVE=1 required")
    if not key: raise RuntimeError("API key required")
    if not (1 <= concurrency <= 8): raise ValueError("concurrency must be 1..8")
    plan=json.loads(Path(plan_path).read_text()); frozen=validate_plan(plan)
    ledger_file=Path(ledger_path)
    if not ledger_file.exists(): raise FileNotFoundError("existing ledger required")
    ledger=PersistentBudget(ledger_file,cap)
    out=Path(output)
    if out.exists(): raise FileExistsError(str(out))
    out.mkdir(parents=True); (out/"frozen_plan.json").write_bytes(Path(plan_path).read_bytes())
    import httpx
    owned=client is None
    if owned:
        client=httpx.Client(base_url="https://api.typesafe.ai",headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},timeout=120)
    red=Redactor([key] if key else [])
    rows=out/"attempts.jsonl"; lock=__import__("threading").Lock(); summary={"planned":len(frozen),"attempted":0,"unsent":0,"errors":0,"stopped":False}
    try:
        for rid,body,payload in frozen:
            try: reservation=ledger.reserve(rid,MAX_INPUT*RATE,label="expanded-dispatch")
            except Exception as e:
                summary["stopped"]=True; summary["unsent"]=len(frozen)-summary["attempted"]; break
            rec={"attempt_id":str(uuid.uuid4()),"request_id":rid,"request_sha256":hashlib.sha256(body).hexdigest(),"started":time.time()}
            t=time.monotonic()
            try:
                resp=client.post("/v1/systemone",content=body); raw=bytes(resp.content)
                rec.update({"status":resp.status_code,"response_sha256":hashlib.sha256(raw).hexdigest(),"raw":red.text(raw.decode("utf-8","replace")),"headers":red.headers(dict(resp.headers)),"elapsed_ms":(time.monotonic()-t)*1000})
                try: parsed=json.loads(raw); rec["body"]=red.obj(parsed)
                except Exception as e: parsed=None; rec["parse_error"]=red.text(str(e))
                usage=parsed.get("usage") if isinstance(parsed,dict) else None; n=usage.get("input_tokens") if isinstance(usage,dict) else None
                known=isinstance(n,int) and not isinstance(n,bool) and 0<=n<=MAX_INPUT
                model=parsed.get("model") if isinstance(parsed,dict) else None
                ans=parsed.get("answers",{}).get("q0") if isinstance(parsed,dict) and isinstance(parsed.get("answers"),dict) else None
                valid_answer=(isinstance(ans,dict) and ans.get("type")=="choice" and ans.get("choice") in payload["questions"]["q0"]["criteria"] and isinstance(ans.get("probabilities"),dict) and set(ans["probabilities"])==set(payload["questions"]["q0"]["criteria"]) and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in ans["probabilities"].values()))
                if resp.status_code==200 and model=="jev-1.13.0" and known and valid_answer: ledger.settle(reservation,n*RATE)
                else: ledger.settle(reservation,None); summary["errors"]+=1; summary["stopped"]=True
            except Exception as e:
                rec.update({"status":0,"error":red.exception(e),"elapsed_ms":(time.monotonic()-t)*1000}); ledger.settle(reservation,None); summary["errors"]+=1; summary["stopped"]=True
            with lock, rows.open("a",encoding="utf8") as f: f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
            summary["attempted"]+=1
            if summary["stopped"]: summary["unsent"]+=len(frozen)-summary["attempted"]; break
    finally:
        if owned: client.close()
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); return summary
