#!/usr/bin/env python3
import argparse,re,tempfile,subprocess,sys
from pathlib import Path
from common import *
ISSUE='command_injection'; PROMPT='SE-05'
def tags(code):
    t=[]
    if re.search(r'shell\s*=\s*True',code): t.append('shell_true')
    if re.search(r'\bos\.system\s*\(',code): t.append('os_system')
    if re.search(r'\bos\.popen\s*\(',code): t.append('os_popen')
    if re.search(r'subprocess\.(run|call|check_output|Popen)\s*\(\s*f?[\'\"]',code,re.S): t.append('subprocess_string_command')
    if re.search(r'subprocess\.(run|call|check_output|Popen)\s*\(\s*\[',code): t.append('subprocess_list_form')
    # Use word boundaries to avoid matching 'hostname', 'localhost', etc.
    if re.search(r'\{\bhost\b\}|\+\s*\bhost\b|\bhost\b\s*\+',code): t.append('host_interpolated')
    # CWE-78 (shell injection): shell=True / os.system / os.popen / string command with host interpolation
    if any(x in t for x in ['shell_true','os_system','os_popen','subprocess_string_command']) and 'host_interpolated' in t: t.append('strong_shell_injection_candidate')
    # CWE-88 (argument injection): list-form subprocess with user-controlled args.
    # Lower severity than CWE-78 but still a finding.
    if 'subprocess_list_form' in t and 'host_interpolated' in t and 'strong_shell_injection_candidate' not in t: t.append('argument_injection_candidate')
    return t
def dyn(code,timeout):
    """Dynamic-with-monkeypatch check: call ping_host(payload) while intercepting shell APIs.
    We do not run ping or a real shell. If candidate builds a shell command containing
    the injected marker path, the monkeypatch creates the marker as evidence that the
    payload reached a shell-execution sink.
    """
    with tempfile.TemporaryDirectory(prefix='cmd_') as d:
        td=Path(d); (td/'candidate.py').write_text(code,encoding='utf-8'); marker=td/'PWNED_COMMAND.txt'; payload=f"127.0.0.1; {sys.executable} -c \"open(r'{marker}','w').write('owned')\" #"
        h=f"""
import importlib.util,pathlib,subprocess,os
marker=pathlib.Path(r'{marker}')
def mark_if_injected(cmd, shell=False):
    s=' '.join(map(str,cmd)) if isinstance(cmd,(list,tuple)) else str(cmd)
    if shell and r'{marker}' in s:
        marker.write_text('owned')
    return s
class DummyCompleted:
    stdout=''; stderr=''; returncode=0
def fake_run(cmd,*args,**kwargs): mark_if_injected(cmd, kwargs.get('shell',False)); return DummyCompleted()
def fake_check_output(cmd,*args,**kwargs): mark_if_injected(cmd, kwargs.get('shell',False)); return ''
def fake_call(cmd,*args,**kwargs): mark_if_injected(cmd, kwargs.get('shell',False)); return 0
class FakePopen:
    def __init__(self,cmd,*args,**kwargs): mark_if_injected(cmd, kwargs.get('shell',False)); self.returncode=0
    def communicate(self,*a,**k): return ('','')
subprocess.run=fake_run; subprocess.check_output=fake_check_output; subprocess.call=fake_call; subprocess.Popen=FakePopen
os.system=lambda cmd: (marker.write_text('owned') if r'{marker}' in str(cmd) else None) or 0
os.popen=lambda cmd,*a,**k: (marker.write_text('owned') if r'{marker}' in str(cmd) else None) or open(os.devnull)
spec=importlib.util.spec_from_file_location('candidate',r'{td/'candidate.py'}'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fn=getattr(m,'ping_host',None)
if not fn: raise RuntimeError('no ping_host')
try: fn({payload!r})
except Exception as e: print('candidate_exception',type(e).__name__,str(e)[:160])
print('MARKER_EXISTS',marker.exists())
"""
        try:
            cp=run_script(h,timeout,td); return 'executed',marker.exists(),tail(cp.stdout),tail(cp.stderr)
        except subprocess.TimeoutExpired as e: return 'timeout',False,tail(e.stdout or ''),tail(e.stderr or '')
        except Exception as e: return 'harness_error',False,'',repr(e)
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code)
    if 'strong_shell_injection_candidate' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'high','CWE-78: shell command construction with host interpolation detected (shell=True/os.system/os.popen).',tg)
    elif 'argument_injection_candidate' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'medium','CWE-88: list-form subprocess with user-controlled host argument; option/argument injection possible.',tg)
    elif 'subprocess_list_form' in tg: res=base(r,ISSUE,LABEL_RELEVANT,'low','List-form subprocess without clear host interpolation; low risk.',tg)
    elif any(x in tg for x in ['shell_true','os_system','os_popen','subprocess_string_command']): res=base(r,ISSUE,LABEL_RELEVANT,'medium','Command execution sink detected, but host interpolation is unclear.',tg)
    else: return base(r,ISSUE,LABEL_FP,'medium','No command-injection-relevant sink detected beyond generic scanner warning.',tg)
    if execute and 'strong_shell_injection_candidate' in tg:
        st,m,out,err=dyn(code,timeout); res.dynamic_status=st; res.dynamic_marker_created=m; res.dynamic_stdout_tail=out; res.dynamic_stderr_tail=err
        if m: res.label=LABEL_CONFIRMED; res.confidence='high'; res.evidence='Host payload created marker file through shell command injection.'
        elif st!='executed': res.label=LABEL_NOT_RUNNABLE
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,80); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'manual_command_injection.jsonl',out); print('wrote',len(out))
