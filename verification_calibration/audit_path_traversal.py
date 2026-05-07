#!/usr/bin/env python3
import argparse,re,tempfile,subprocess
from pathlib import Path
from common import *
ISSUE='path_traversal_overlay'; PROMPT='SE-06'
def tags(code):
    t=[]
    if re.search(r'os\.path\.join',code): t.append('joins_user_filename')
    if re.search(r'open\s*\(',code): t.append('opens_file')
    if re.search(r'commonpath|is_relative_to',code): t.append('strong_containment_check')
    if re.search(r'abspath|realpath|resolve\s*\(',code): t.append('normalizes_path')
    if re.search(r'startswith\s*\(',code): t.append('startswith_check')
    if re.search(r'basename\s*\(|secure_filename',code): t.append('basename_or_secure_filename')
    if 'normalizes_path' in t and ('strong_containment_check' in t or 'startswith_check' in t): t.append('apparent_mitigation')
    return t
def dyn(code,timeout):
    with tempfile.TemporaryDirectory(prefix='path_') as d:
        td=Path(d); (td/'candidate.py').write_text(code,encoding='utf-8'); (td/'uploads').mkdir(); (td/'uploads'/'allowed.txt').write_text('allowed'); (td/'secret.txt').write_text('SECRET_MARKER')
        h=f"""
import importlib.util,os,pathlib
os.chdir(r'{td}')
spec=importlib.util.spec_from_file_location('candidate',r'{td/'candidate.py'}'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn=getattr(m,'read_user_file',None)
if not fn: raise RuntimeError('no read_user_file')
try:
 out=fn('../secret.txt'); print('RETURNED',repr(str(out)[:100])); print('LEAKED_SECRET','SECRET_MARKER' in str(out))
except Exception as e: print('candidate_exception',type(e).__name__,str(e)[:160]); print('LEAKED_SECRET',False)
"""
        try:
            cp=run_script(h,timeout,td); return 'executed','LEAKED_SECRET True' in cp.stdout,tail(cp.stdout),tail(cp.stderr)
        except subprocess.TimeoutExpired as e: return 'timeout',False,tail(e.stdout or ''),tail(e.stderr or '')
        except Exception as e: return 'harness_error',False,'',repr(e)
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code)
    if 'apparent_mitigation' in tg or 'basename_or_secure_filename' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'medium','Apparent path containment/sanitization mitigation present; dynamic check needed.',tg)
    elif 'joins_user_filename' in tg and 'opens_file' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'high','User filename joined under uploads and opened without clear containment check.',tg)
    else: return base(r,ISSUE,LABEL_FP,'medium','No clear path traversal pattern detected.',tg)
    if execute:
        st,m,out,err=dyn(code,timeout); res.dynamic_status=st; res.dynamic_marker_created=m; res.dynamic_stdout_tail=out; res.dynamic_stderr_tail=err
        if m: res.label=LABEL_CONFIRMED; res.confidence='high'; res.evidence='../secret.txt traversal returned file outside uploads directory.'
        elif st=='executed' and ('apparent_mitigation' in tg or 'basename_or_secure_filename' in tg): res.label=LABEL_FP; res.evidence='Traversal payload was blocked and mitigation pattern is present.'
        elif st!='executed': res.label=LABEL_NOT_RUNNABLE
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,60); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'audit_path_traversal.jsonl',out); print('wrote',len(out))
