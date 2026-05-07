#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
from common import *
ORDER=[LABEL_CONFIRMED,LABEL_RELEVANT,LABEL_FP,LABEL_NOT_RUNNABLE,LABEL_OUT_OF_SCOPE]
EXP={'unsafe_deserialization','unsafe_eval','command_injection'}; AUD={'path_traversal_overlay','mass_assignment_overlay','sql_injection_residual_audit'}
def load_all(inputs):
    rows=[]
    for inp in inputs:
        p=Path(inp); files=sorted(p.glob('*.jsonl')) if p.is_dir() else [p]
        for f in files:
            if f.name.startswith(('manual_','audit_')): rows+=load_jsonl(f)
    return rows
def pct(n,d): return 0 if d==0 else 100*n/d
def counts(rows):
    d={k:0 for k in ORDER}
    for r in rows: d[r.get('label','')]=d.get(r.get('label',''),0)+1
    return d
def summary(name,rows):
    c=counts(rows); n=len(rows); runnable=n-c.get(LABEL_NOT_RUNNABLE,0)
    return {'group':name,'n':n,'runnable_n':runnable,**{k:c.get(k,0) for k in ORDER},'confirmed_pct_all':pct(c[LABEL_CONFIRMED],n),'relevant_pct_all':pct(c[LABEL_RELEVANT],n),'fp_pct_all':pct(c[LABEL_FP],n),'confirmed_pct_runnable':pct(c[LABEL_CONFIRMED],runnable)}
def group(rows,k):
    d={}
    for r in rows: d.setdefault(str(r.get(k,'')),[]).append(r)
    return d
def write_csv(p,rows):
    if not rows: return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--exclude-not-runnable-from-headline',action='store_true'); a=ap.parse_args(); rows=load_all(a.inputs); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    exploit=[r for r in rows if r.get('issue_class') in EXP]; audit=[r for r in rows if r.get('issue_class') in AUD]
    es=summary('exploit_verification',exploit); c=counts(exploit); denom=es['runnable_n'] if a.exclude_not_runnable_from_headline else es['n']
    phrase='runnable manually reviewed critical calibrated findings' if a.exclude_not_runnable_from_headline else 'manually reviewed critical calibrated findings'
    sent=f"Across {denom} {phrase}, {pct(c[LABEL_CONFIRMED],denom):.1f}% were confirmed exploitable, {pct(c[LABEL_RELEVANT],denom):.1f}% were security-relevant but not directly exploitable, and {pct(c[LABEL_FP],denom):.1f}% were false positives."
    by_issue=[summary(k,v) for k,v in sorted(group(rows,'issue_class').items())]; by_model=[summary(k,v) for k,v in sorted(group(rows,'model_display').items())]; by_prompt=[summary(k,v) for k,v in sorted(group(rows,'prompt_id').items())]
    obj={'headline_sentence':sent,'all':summary('all',rows),'exploit_verification':es,'calibration_audit':summary('calibration_audit',audit),'by_issue':by_issue,'by_model':by_model,'by_prompt':by_prompt}
    write_json(out/'manual_verification_summary.json',obj); write_csv(out/'by_issue.csv',by_issue); write_csv(out/'by_model.csv',by_model); write_csv(out/'by_prompt.csv',by_prompt)
    (out/'summary.md').write_text('# Manual verification summary\n\n'+sent+'\n\n## By issue class\n'+'\n'.join(f"- **{r['group']}**: n={r['n']}, confirmed={r[LABEL_CONFIRMED]}, relevant={r[LABEL_RELEVANT]}, FP={r[LABEL_FP]}, not_runnable={r[LABEL_NOT_RUNNABLE]}" for r in by_issue)+'\n',encoding='utf-8')
    print(sent); print('wrote report to',out)
