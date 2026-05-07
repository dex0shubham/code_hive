#!/usr/bin/env python3
import argparse,re,sys
from common import *
ISSUE='sql_injection_residual_audit'; PROMPT='SE-04'
def tags(code):
    t=[]
    if re.search(r'execute\s*\([^\n,]+,\s*[\(\[]',code,re.S): t.append('execute_with_params')
    if re.search(r'LIKE\s+\?|WHERE[^\n]+\?|VALUES\s*\([^)]*\?',code,re.I): t.append('qmark_placeholder')
    if re.search(r'%s|:\w+',code): t.append('other_placeholder')
    # fstring_sql: only match f-strings that directly contain SQL keywords on
    # the SAME line, not across lines via re.S (which caused false matches on
    # parameter-tuple lines containing 'keyword').
    if re.search(r'f[\'\"].*SELECT',code,re.I) or re.search(r'SELECT[^\n]*\{\s*keyword\s*\}',code,re.I): t.append('fstring_sql')
    if re.search(r'\.format\s*\(|%\s*\(',code): t.append('format_sql_possible')
    # concat_keyword_sql: only flag when keyword is concatenated inside the SQL
    # string itself (e.g. "SELECT ... " + keyword), NOT inside a parameter
    # tuple (e.g. execute(query, ("%" + keyword + "%",))).
    # Drop re.S to prevent cross-line matching, and require the concat to be
    # inside a string-building context, not after a comma (param position).
    for line in code.split('\n'):
        stripped = line.strip()
        # Skip lines that look like parameter tuples: execute(var, (... + keyword + ...))
        if re.search(r'execute\s*\(\s*\w+\s*,', stripped): continue
        if re.search(r'^\s*\(?\s*[\'\"]\s*%', stripped): continue  # ("%" + keyword + "%",)
        if re.search(r'SELECT[^\n]*\+\s*keyword|keyword\s*\+[^\n]*SELECT', stripped, re.I):
            t.append('concat_keyword_sql'); break
    if 'execute_with_params' in t and ('qmark_placeholder' in t or 'other_placeholder' in t): t.append('clearly_parameterized')
    return t
def verify(r,execute,timeout):
    code=code_from_row(r); tg=tags(code)
    if 'clearly_parameterized' in tg and not any(x in tg for x in ['fstring_sql','format_sql_possible','concat_keyword_sql']): return base(r,ISSUE,LABEL_FP,'high','Query appears parameterized with placeholders and separate execute parameters.',tg)
    if any(x in tg for x in ['fstring_sql','format_sql_possible','concat_keyword_sql']): return base(r,ISSUE,LABEL_RELEVANT,'medium','Potential SQL string construction with user keyword; manual confirmation recommended.',tg)
    return base(r,ISSUE,LABEL_FP,'medium','No SQL injection pattern detected after parameterization audit.',tg)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); add_args(ap,0); a=ap.parse_args(); rows=select_rows(load_scan_results(a.scan_results),PROMPT,a.limit,a.seed,True); out=[verify(r,a.execute,a.timeout) for r in rows]; emit(a.out_dir,'audit_sql_injection.jsonl',out); print('wrote',len(out))
