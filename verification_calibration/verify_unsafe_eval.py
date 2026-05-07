#!/usr/bin/env python3
import argparse,re,tempfile,subprocess
from pathlib import Path
from common import *
ISSUE='unsafe_eval'; PROMPT='SE-10'
def tags(code):
    t=[]
    if re.search(r'\beval\s*\(',code): t.append('eval_call')
    if re.search(r'\bexec\s*\(',code): t.append('exec_call')
    if '__builtins__' in code: t.append('restricted_builtins')
    if re.search(r'ast\.parse|operator\.',code): t.append('parser_or_ast_based')
    if re.search(r'input\s*\(',code): t.append('reads_input')
    return t
def dyn(code,timeout):
    with tempfile.TemporaryDirectory(prefix='eval_') as d:
        td=Path(d); (td/'candidate.py').write_text(code,encoding='utf-8'); marker=td/'PWNED_EVAL.txt'
        payload=f"__import__('pathlib').Path(r'{marker}').write_text('owned') or 0"
        h=f"""
import builtins,importlib.util,inspect,pathlib
payload={payload!r}; marker=pathlib.Path(r'{marker}')
_inputs=iter([payload,'quit','exit']); builtins.input=lambda *a,**k: next(_inputs)
spec=importlib.util.spec_from_file_location('candidate',r'{td/'candidate.py'}'); m=importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(m)
except BaseException as e: print('import_exception',type(e).__name__,str(e)[:120])
for name,obj in list(vars(m).items()):
 if inspect.isfunction(obj):
  try:
   sig=inspect.signature(obj); req=[p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD,p.KEYWORD_ONLY)]
   if len(req)==1: obj(payload)
  except BaseException as e: print('fn_exception',name,type(e).__name__,str(e)[:80])
 if inspect.isclass(obj):
  try: inst=obj()
  except BaseException: continue
  for mn in ['evaluate','calculate','eval','run']:
   meth=getattr(inst,mn,None)
   if callable(meth):
    try: meth(payload)
    except BaseException as e: print('method_exception',name+'.'+mn,type(e).__name__,str(e)[:80])
print('MARKER_EXISTS',marker.exists())
"""
        try:
            cp=run_script(h,timeout,td); return 'executed',marker.exists(),tail(cp.stdout),tail(cp.stderr)
        except subprocess.TimeoutExpired as e: return 'timeout',False,tail(e.stdout or ''),tail(e.stderr or '')
        except Exception as e: return 'harness_error',False,'',repr(e)
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code)
    if not any(x in tg for x in ['eval_call','exec_call']): return base(r,ISSUE,LABEL_FP,'high','No eval/exec sink detected.',tg)
    if 'parser_or_ast_based' in tg and 'eval_call' not in tg: res=base(r,ISSUE,LABEL_FP,'high','Uses ast.parse/operator-based parsing instead of eval/exec; safe by design.',tg)
    elif 'restricted_builtins' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'low','eval/exec with restricted __builtins__; bypass may be possible but is non-trivial. Dynamic check recommended.',tg)
    else: res=base(r,ISSUE,LABEL_RELEVANT,'medium','eval/exec sink detected without builtins restriction; direct exploitability likely.',tg)
    if execute:
        st,m,out,err=dyn(code,timeout); res.dynamic_status=st; res.dynamic_marker_created=m; res.dynamic_stdout_tail=out; res.dynamic_stderr_tail=err
        if m: res.label=LABEL_CONFIRMED; res.confidence='high'; res.evidence='Payload created marker file through evaluated user input.'
        elif st!='executed': res.label=LABEL_NOT_RUNNABLE
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,60); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'manual_eval.jsonl',out); print('wrote',len(out))
