#!/usr/bin/env python3
import argparse,re,tempfile,subprocess
from pathlib import Path
from common import *
ISSUE='unsafe_deserialization'; PROMPT='SE-08'
def tags(code):
    t=[]
    if re.search(r'\bpickle\.loads?\s*\(',code): t.append('pickle_load')
    if re.search(r'\bjoblib\.load\s*\(',code): t.append('joblib_load')
    if re.search(r'\btorch\.load\s*\(',code): t.append('torch_load')
    if 'def load_model' in code: t.append('has_load_model')
    return t
def dyn(code,timeout):
    with tempfile.TemporaryDirectory(prefix='deser_') as d:
        td=Path(d); (td/'candidate.py').write_text(code,encoding='utf-8'); marker=td/'PWNED_DESER.txt'; mal=td/'mal.pkl'
        h=f"""
import pickle,pathlib,importlib.util
class E:
 def __reduce__(self): return (pathlib.Path(r'{marker}').write_text,('owned',))
with open(r'{mal}','wb') as f: pickle.dump(E(),f)
spec=importlib.util.spec_from_file_location('candidate',r'{td/'candidate.py'}'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn=getattr(m,'load_model',None)
if not fn: raise RuntimeError('no load_model')
try: fn(r'{mal}')
except Exception as e: print('candidate_exception',type(e).__name__,str(e)[:200])
print('MARKER_EXISTS',pathlib.Path(r'{marker}').exists())
"""
        try:
            cp=run_script(h,timeout,td); return 'executed',marker.exists(),tail(cp.stdout),tail(cp.stderr)
        except subprocess.TimeoutExpired as e: return 'timeout',False,tail(e.stdout or ''),tail(e.stderr or '')
        except Exception as e: return 'harness_error',False,'',repr(e)
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code); risky=any(x in tg for x in ['pickle_load','joblib_load','torch_load'])
    if not risky: return base(r,ISSUE,LABEL_FP,'high','No pickle/joblib/torch load sink detected.',tg)
    res=base(r,ISSUE,LABEL_RELEVANT,'high','Unsafe Python-object deserialization sink detected.',tg)
    if execute:
        st,m,out,err=dyn(code,timeout); res.dynamic_status=st; res.dynamic_marker_created=m; res.dynamic_stdout_tail=out; res.dynamic_stderr_tail=err
        if m: res.label=LABEL_CONFIRMED; res.evidence='Malicious pickle created marker file when load_model(path) was called.'
        elif st!='executed': res.label=LABEL_NOT_RUNNABLE; res.evidence='Sink detected but automated harness could not execute it.'
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,50); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'manual_deserialization.jsonl',out); print('wrote',len(out))
