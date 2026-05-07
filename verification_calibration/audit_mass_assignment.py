#!/usr/bin/env python3
import argparse,re,tempfile,subprocess
from pathlib import Path
from common import *
ISSUE='mass_assignment_overlay'; PROMPT='SE-28'
def tags(code):
    t=[]
    if re.search(r'\.items\s*\(\s*\)',code): t.append('payload_items_loop')
    if re.search(r'setattr\s*\(',code): t.append('setattr')
    if re.search(r'allowed_?fields|editable_?fields|PERMITTED|WHITELIST|valid_?fields|safe_?fields',code,re.I): t.append('allowlist_present')
    if re.search(r'field\s+in\s+\{|key\s+in\s+\{',code): t.append('inline_allowlist_set')
    if re.search(r'is_admin|role|permission',code,re.I): t.append('mentions_sensitive_fields')
    # hasattr(obj, key) is NOT a mass-assignment guard: privileged fields
    # like is_admin/role/permissions are legitimate attributes that hasattr
    # confirms exist. Only explicit allowlist membership checks count.
    if re.search(r'hasattr\s*\(\s*\w+\s*,\s*\w+\s*\)',code) and 'allowlist_present' not in t and 'inline_allowlist_set' not in t: t.append('hasattr_only_guard')
    if 'payload_items_loop' in t and 'setattr' in t and 'allowlist_present' not in t and 'inline_allowlist_set' not in t: t.append('unrestricted_setattr_candidate')
    return t
def dyn(code,timeout):
    with tempfile.TemporaryDirectory(prefix='mass_') as d:
        td=Path(d); (td/'candidate.py').write_text(code,encoding='utf-8')
        h=f"""
import importlib.util,inspect
class Session:
 def commit(self): pass
 def rollback(self): pass
class User:
 def __init__(self): self.email='a@x'; self.is_admin=False; self.role='user'; self.permissions=[]
u=User(); payload={{'email':'b@x','is_admin':True,'role':'admin','permissions':['*']}}
spec=importlib.util.spec_from_file_location('candidate',r'{td/'candidate.py'}'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn=getattr(m,'update_profile',None)
if not fn: raise RuntimeError('no update_profile')
try:
 params=list(inspect.signature(fn).parameters)
 if len(params)>=3: fn(Session(),u,payload)
 elif len(params)==2: fn(u,payload)
 else: raise RuntimeError('unsupported signature')
except Exception as e: print('candidate_exception',type(e).__name__,str(e)[:160])
print('USER_STATE',u.is_admin,u.role,u.permissions); print('PRIV_ESCALATED',bool(u.is_admin or u.role=='admin' or u.permissions))
"""
        try:
            cp=run_script(h,timeout,td); return 'executed','PRIV_ESCALATED True' in cp.stdout,tail(cp.stdout),tail(cp.stderr)
        except subprocess.TimeoutExpired as e: return 'timeout',False,tail(e.stdout or ''),tail(e.stderr or '')
        except Exception as e: return 'harness_error',False,'',repr(e)
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code)
    if 'unrestricted_setattr_candidate' in tg and 'hasattr_only_guard' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'high','Payload items assigned via setattr with only hasattr() guard (not a real allowlist — hasattr confirms privileged fields like is_admin exist).',tg)
    elif 'unrestricted_setattr_candidate' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'high','Payload items assigned via setattr without any field guard.',tg)
    elif 'setattr' in tg and ('allowlist_present' in tg or 'inline_allowlist_set' in tg): res=base(r,ISSUE,LABEL_RELEVANT,'medium','setattr loop appears guarded by allowlist; dynamic check needed to confirm guard effectiveness.',tg)
    elif 'setattr' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'medium','setattr detected; manual review needed.',tg)
    else: return base(r,ISSUE,LABEL_FP,'medium','No setattr-based mass-assignment pattern detected.',tg)
    if execute:
        st,m,out,err=dyn(code,timeout); res.dynamic_status=st; res.dynamic_marker_created=m; res.dynamic_stdout_tail=out; res.dynamic_stderr_tail=err
        if m: res.label=LABEL_CONFIRMED; res.confidence='high'; res.evidence='Payload changed protected user attributes such as is_admin/role/permissions.'
        elif st=='executed' and 'allowlist_present' in tg: res.label=LABEL_FP; res.evidence='Protected-field payload was blocked and allowlist pattern is present.'
        elif st!='executed': res.label=LABEL_NOT_RUNNABLE
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,60); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'audit_mass_assignment.jsonl',out); print('wrote',len(out))
