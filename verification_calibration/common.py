from __future__ import annotations
import argparse, ast, csv, hashlib, json, os, random, re, subprocess, sys, tempfile, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
LABEL_CONFIRMED='confirmed_exploitable'; LABEL_RELEVANT='security_relevant_not_directly_exploitable'; LABEL_FP='false_positive'; LABEL_NOT_RUNNABLE='not_runnable'; LABEL_OUT_OF_SCOPE='out_of_scope'
SEED_DEFAULT=20260506
@dataclass
class ManualResult:
    prompt_id:str; model_display:str; temperature:float|str; sample_index:int|str; issue_class:str; label:str; confidence:str; evidence:str; detector_findings:list; cwe_set:list; static_tags:list; dynamic_status:str='not_attempted'; dynamic_marker_created:bool=False; dynamic_stdout_tail:str=''; dynamic_stderr_tail:str=''; code_sha256:str=''; notes:str=''
def tail(s,n=1200): return (s or '')[-n:]
def code_from_row(r): return r.get('_code') or r.get('response_text') or ''
def sha(s): return hashlib.sha256(code_from_row(s).encode('utf-8','replace')).hexdigest() if isinstance(s,dict) else hashlib.sha256(str(s).encode()).hexdigest()
def load_jsonl(p):
    out=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out
def load_scan_results(path):
    p=Path(path); rows=[]
    if p.is_dir():
        hits=list(p.rglob('scan_results.jsonl'))
        if not hits: raise FileNotFoundError('scan_results.jsonl not found')
        return load_jsonl(hits[0])
    if p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if n.endswith('scan_results.jsonl')]
            if not names: raise FileNotFoundError('scan_results.jsonl not found in zip')
            with z.open(names[0]) as f:
                for b in f:
                    if b.strip(): rows.append(json.loads(b.decode('utf-8')))
        return rows
    return load_jsonl(p)
def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for r in rows:
            if hasattr(r,'__dataclass_fields__'): r=asdict(r)
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True),encoding='utf-8')
def select_rows(rows,prompt,limit,seed,only_vulnerable=True):
    pool=[r for r in rows if r.get('prompt_id')==prompt and (not only_vulnerable or r.get('is_vulnerable')) and code_from_row(r).strip()]
    if not limit or limit<=0: return sorted(pool,key=lambda r:(str(r.get('model_display')),str(r.get('temperature')),int(r.get('sample_index',0))))
    by={}
    for r in pool: by.setdefault(str(r.get('model_display')),[]).append(r)
    rng=random.Random(seed)
    for v in by.values(): rng.shuffle(v)
    out=[]; keys=sorted(by); i=0
    while len(out)<limit and any(by.values()):
        k=keys[i%len(keys)]
        if by[k]: out.append(by[k].pop())
        i+=1
    return out[:limit]
def base(r,issue,label,conf,evidence,tags):
    return ManualResult(str(r.get('prompt_id')),str(r.get('model_display')),r.get('temperature',''),r.get('sample_index',''),issue,label,conf,evidence,r.get('findings',[]),list(r.get('cwe_set',[])),tags,code_sha256=sha(r))
def add_args(ap,default_limit):
    ap.add_argument('--scan-results',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--limit',type=int,default=default_limit); ap.add_argument('--seed',type=int,default=SEED_DEFAULT); ap.add_argument('--execute',action='store_true'); ap.add_argument('--timeout',type=float,default=4.0)
def run_script(script,timeout,cwd):
    return subprocess.run([sys.executable,'-I','-c',script],cwd=str(cwd),text=True,capture_output=True,timeout=timeout,env={'PYTHONIOENCODING':'utf-8','PATH':os.environ.get('PATH','')})
def emit(out_dir,name,results):
    write_jsonl(Path(out_dir)/name,results)
    counts={}
    for r in results: counts[r.label]=counts.get(r.label,0)+1
    write_json(Path(out_dir)/name.replace('.jsonl','.summary.json'),{'n':len(results),'label_counts':counts})
    if not results:
        print(f"WARNING: 0 samples written to {name}. If this is an audit "
              f"script (SE-06/SE-28), did you pass the CALIBRATED "
              f"scan_results.jsonl? Raw CodeShield data has 0 findings for "
              f"these prompts.", file=sys.stderr)
