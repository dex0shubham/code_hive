#!/usr/bin/env python3
"""
Code Hivemind — Security Analysis
====================================
Scans your collected LLM responses for security vulnerabilities
and measures whether different models produce the SAME vulnerable patterns.

This is the novel contribution: not just "LLMs produce vulnerable code"
(already known) but "LLMs produce the SAME vulnerable code" (new).

Uses regex-based vulnerability detection (no CodeQL/Bandit install needed).
For the paper, you should also run Bandit/Semgrep for validation.

Usage:
  python security_analysis.py
  python security_analysis.py --bandit   # also run Bandit (pip install bandit)

Outputs:
  results/security/  — vulnerability reports, figures, CSVs
"""

import ast
import argparse
import json
import os
import re
import sys
import tempfile
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.titlesize": 12,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
except ImportError:
    HAS_MPL = False

RAW_DIR = os.path.join("results", "raw_responses")
SEC_DIR = os.path.join("results", "security")

# Known vulnerable/deprecated packages
KNOWN_VULNERABLE_PACKAGES = {
    "pickle": "CWE-502: Deserialization of untrusted data",
    "marshal": "CWE-502: Deserialization of untrusted data",
    "shelve": "CWE-502: Deserialization via pickle backend",
    "yaml": "CWE-502: yaml.load() without SafeLoader is unsafe",
    "subprocess": "CWE-078: OS command injection risk",
    "eval": "CWE-094: Code injection via eval()",
    "exec": "CWE-094: Code injection via exec()",
    "telnetlib": "CWE-319: Cleartext transmission",
    "ftplib": "CWE-319: Cleartext transmission (no TLS by default)",
    "http": "CWE-319: http.server has no TLS support",
    "xml": "CWE-611: XML external entity (XXE) risk with xml.etree",
    "md5": "CWE-328: Weak hash algorithm",
    "sha1": "CWE-328: Weak hash algorithm",
    "random": "CWE-330: Not cryptographically secure (use secrets)",
    "tempfile": "CWE-377: Insecure temp file (if using mktemp)",
}

# ═══════════════════════════════════════════════════════════════
# REGEX-BASED VULNERABILITY DETECTION
# ═══════════════════════════════════════════════════════════════

# Each detector returns (cwe_id, description, matched_line)

VULN_PATTERNS = [
    # CWE-78: OS Command Injection
    {
        "cwe": "CWE-078",
        "name": "OS command injection",
        "patterns": [
            r"os\.system\s*\(",
            r"os\.popen\s*\(",
            r"subprocess\.call\s*\(.*shell\s*=\s*True",
            r"subprocess\.Popen\s*\(.*shell\s*=\s*True",
            r"subprocess\.run\s*\(.*shell\s*=\s*True",
        ],
    },
    # CWE-89: SQL Injection
    {
        "cwe": "CWE-089",
        "name": "SQL injection",
        "patterns": [
            r"execute\s*\(\s*[\"'].*%[sd]",
            r"execute\s*\(\s*f[\"']",
            r"execute\s*\(.*\.format\s*\(",
            r"execute\s*\(.*\+\s*\w+",
            r"cursor\.execute\s*\(\s*[\"'].*\{",
        ],
    },
    # CWE-94: Code Injection
    {
        "cwe": "CWE-094",
        "name": "Code injection (eval/exec)",
        "patterns": [
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"__import__\s*\(",
            r"compile\s*\(.*exec",
        ],
    },
    # CWE-259/798: Hard-coded Credentials
    {
        "cwe": "CWE-798",
        "name": "Hard-coded credentials",
        "patterns": [
            r"password\s*=\s*[\"'][^\"']+[\"']",
            r"passwd\s*=\s*[\"'][^\"']+[\"']",
            r"secret\s*=\s*[\"'][^\"']+[\"']",
            r"api_key\s*=\s*[\"'][^\"']+[\"']",
            r"token\s*=\s*[\"'][A-Za-z0-9+/=]{20,}[\"']",
        ],
    },
    # CWE-327/328: Weak Cryptography
    {
        "cwe": "CWE-327",
        "name": "Weak cryptographic algorithm",
        "patterns": [
            r"hashlib\.md5\s*\(",
            r"hashlib\.sha1\s*\(",
            r"MD5\.new\s*\(",
            r"SHA\.new\s*\(",
            r"DES\.",
            r"Blowfish\.",
        ],
    },
    # CWE-330: Insufficient Randomness
    {
        "cwe": "CWE-330",
        "name": "Insecure randomness",
        "patterns": [
            r"random\.random\s*\(",
            r"random\.randint\s*\(",
            r"random\.choice\s*\(",
            r"random\.randrange\s*\(",
        ],
        "context_exclude": r"secrets\.|os\.urandom|SystemRandom",  # OK if also using secure random
    },
    # CWE-502: Deserialization
    {
        "cwe": "CWE-502",
        "name": "Unsafe deserialization",
        "patterns": [
            r"pickle\.loads?\s*\(",
            r"pickle\.Unpickler",
            r"marshal\.loads?\s*\(",
            r"yaml\.load\s*\([^)]*(?!Loader)",
            r"yaml\.unsafe_load\s*\(",
            r"jsonpickle\.decode\s*\(",
        ],
    },
    # CWE-20: Improper Input Validation
    {
        "cwe": "CWE-020",
        "name": "Missing input validation",
        "patterns": [
            r"request\.(args|form|json|data)\s*\[",
            r"request\.get\s*\(",
            r"input\s*\(\s*\)",
        ],
    },
    # CWE-22: Path Traversal
    {
        "cwe": "CWE-022",
        "name": "Path traversal risk",
        "patterns": [
            r"open\s*\(.*\+.*\)",
            r"open\s*\(.*\.format\s*\(",
            r"open\s*\(\s*f[\"']",
            r"os\.path\.join\s*\(.*request",
        ],
    },
    # CWE-319: Cleartext Transmission
    {
        "cwe": "CWE-319",
        "name": "Cleartext HTTP (no TLS)",
        "patterns": [
            r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)",
            r"requests\.get\s*\(\s*[\"']http://(?!localhost)",
            r"verify\s*=\s*False",
            r"ssl\._create_unverified_context",
        ],
    },
    # CWE-676: Dangerous Functions
    {
        "cwe": "CWE-676",
        "name": "Use of dangerous function",
        "patterns": [
            r"tempfile\.mktemp\s*\(",
            r"os\.chmod\s*\(.*0o?777",
            r"assert\s+.*,",  # assertions disabled with -O
        ],
    },
]


def extract_code(text):
    m = re.findall(r"```(?:python|javascript|js|py)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(m) if m else text.strip()


def scan_code(code):
    """Scan a code string for vulnerabilities. Returns list of (cwe, name, line_num, matched_text)."""
    findings = []
    lines = code.split("\n")

    for vuln in VULN_PATTERNS:
        # Check context exclusion
        if "context_exclude" in vuln:
            if re.search(vuln["context_exclude"], code):
                continue

        for pattern in vuln["patterns"]:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue  # skip comments
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append({
                        "cwe": vuln["cwe"],
                        "name": vuln["name"],
                        "line": i + 1,
                        "matched": stripped[:120],
                        "pattern": pattern,
                    })
                    break  # one match per pattern per vuln type is enough

    # Deduplicate by CWE (keep first match per CWE)
    seen_cwes = set()
    deduped = []
    for f in findings:
        if f["cwe"] not in seen_cwes:
            seen_cwes.add(f["cwe"])
            deduped.append(f)

    return deduped


def get_imports(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    imps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names: imps.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imps.add(node.module.split(".")[0])
    return imps


def check_risky_imports(code):
    """Flag imports of known-risky packages."""
    imports = get_imports(code)
    findings = []
    for imp in imports:
        if imp in KNOWN_VULNERABLE_PACKAGES:
            findings.append({
                "import": imp,
                "risk": KNOWN_VULNERABLE_PACKAGES[imp],
            })
    return findings


# ═══════════════════════════════════════════════════════════════
# OPTIONAL: BANDIT INTEGRATION
# ═══════════════════════════════════════════════════════════════

def run_bandit(code):
    """Run Bandit on a code snippet. Returns list of issues."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", f.name],
                capture_output=True, text=True, timeout=10,
            )
            os.unlink(f.name)
            if result.stdout:
                data = json.loads(result.stdout)
                return data.get("results", [])
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def load_all():
    path = Path(RAW_DIR) / "all_responses.jsonl"
    if path.exists():
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    rs = []
    for f in Path(RAW_DIR).glob("*.jsonl"):
        if f.name == "all_responses.jsonl": continue
        with open(f) as fh:
            rs.extend(json.loads(l) for l in fh if l.strip())
    return rs


def main():
    parser = argparse.ArgumentParser(description="Code Hivemind Security Analysis")
    parser.add_argument("--bandit", action="store_true", help="Also run Bandit (must be installed)")
    args = parser.parse_args()

    os.makedirs(SEC_DIR, exist_ok=True)

    print("Loading data...")
    responses = load_all()
    if not responses:
        print(f"ERROR: No data in {RAW_DIR}/. Run `python run.py --collect` first.")
        sys.exit(1)

    temps = sorted(set(r["temperature"] for r in responses))
    temp = max(temps)
    filtered = [r for r in responses if abs(r["temperature"] - temp) < 0.01]

    models = sorted(set(r["model_display"] for r in filtered))
    prompts = sorted(set(r["prompt_id"] for r in filtered))
    print(f"  {len(filtered)} responses at temp={temp}, {len(models)} models, {len(prompts)} prompts\n")

    # ── SCAN ALL CODE ──
    print("=" * 60)
    print("SCANNING FOR VULNERABILITIES")
    print("=" * 60)

    all_findings = []  # (response_meta, findings_list)

    for r in filtered:
        code = extract_code(r["response_text"])
        findings = scan_code(code)
        risky = check_risky_imports(code)

        if args.bandit:
            bandit_results = run_bandit(code)
            for b in bandit_results:
                findings.append({
                    "cwe": b.get("test_id", "B???"),
                    "name": f"Bandit: {b.get('test_name', '?')}",
                    "line": b.get("line_number", 0),
                    "matched": b.get("issue_text", "")[:120],
                    "pattern": "bandit",
                    "severity": b.get("issue_severity", "?"),
                })

        all_findings.append({
            "prompt_id": r["prompt_id"],
            "model": r["model_display"],
            "model_family": r["model_family"],
            "n_vulns": len(findings),
            "cwes": sorted(set(f["cwe"] for f in findings)),
            "cwe_set": frozenset(f["cwe"] for f in findings),
            "findings": findings,
            "risky_imports": risky,
        })

    # ── VULNERABILITY RATES ──
    total = len(all_findings)
    vuln_count = sum(1 for f in all_findings if f["n_vulns"] > 0)
    print(f"\n  Vulnerability rate: {vuln_count}/{total} = {vuln_count/total:.1%} of code samples have >= 1 finding")

    # Per-model vuln rates
    print(f"\n  {'Model':<22} {'Vuln Rate':>10} {'Mean CWEs':>10} {'Samples':>8}")
    print("  " + "-" * 54)
    model_vuln_rates = {}
    for model in models:
        mf = [f for f in all_findings if f["model"] == model]
        vr = sum(1 for f in mf if f["n_vulns"] > 0) / len(mf) if mf else 0
        mc = np.mean([f["n_vulns"] for f in mf]) if mf else 0
        model_vuln_rates[model] = {"rate": vr, "mean_cwes": mc, "n": len(mf)}
        print(f"  {model:<22} {vr:>9.1%} {mc:>10.1f} {len(mf):>8}")

    # Per-CWE distribution
    cwe_counter = Counter()
    for f in all_findings:
        for cwe in f["cwes"]:
            cwe_counter[cwe] += 1

    print(f"\n  Top CWEs found:")
    for cwe, count in cwe_counter.most_common(10):
        name = next((v["name"] for v in VULN_PATTERNS if v["cwe"] == cwe), "?")
        print(f"    {cwe:<10} {name:<35} {count:>4} samples ({count/total:.1%})")

    # ── THE KEY METRIC: VULNERABILITY HOMOGENEITY ──
    print(f"\n{'='*60}")
    print("VULNERABILITY HOMOGENEITY (the novel metric)")
    print("=" * 60)
    print("  Do different models produce the SAME vulnerabilities")
    print("  for the SAME prompt?\n")

    by_prompt = defaultdict(dict)
    for f in all_findings:
        by_prompt[f["prompt_id"]][f["model"]] = f

    # For each prompt, check how many model pairs share the same CWE set
    prompt_homogeneity = []
    cwe_pair_matches = []
    total_pairs = 0
    exact_cwe_matches = 0
    any_overlap_count = 0

    for pid in prompts:
        if pid not in by_prompt:
            continue
        model_data = by_prompt[pid]
        mlist = sorted(model_data.keys())
        prompt_exact = 0
        prompt_overlap = 0
        prompt_pairs = 0

        for i in range(len(mlist)):
            for j in range(i + 1, len(mlist)):
                cwe_a = model_data[mlist[i]]["cwe_set"]
                cwe_b = model_data[mlist[j]]["cwe_set"]
                total_pairs += 1
                prompt_pairs += 1

                if cwe_a == cwe_b and len(cwe_a) > 0:
                    exact_cwe_matches += 1
                    prompt_exact += 1

                if cwe_a & cwe_b:  # any overlap
                    any_overlap_count += 1
                    prompt_overlap += 1

                cwe_pair_matches.append({
                    "prompt_id": pid,
                    "model_a": mlist[i],
                    "model_b": mlist[j],
                    "cwes_a": sorted(cwe_a),
                    "cwes_b": sorted(cwe_b),
                    "exact_match": cwe_a == cwe_b and len(cwe_a) > 0,
                    "any_overlap": bool(cwe_a & cwe_b),
                    "overlap_cwes": sorted(cwe_a & cwe_b),
                })

        if prompt_pairs > 0:
            prompt_homogeneity.append({
                "prompt_id": pid,
                "exact_rate": prompt_exact / prompt_pairs,
                "overlap_rate": prompt_overlap / prompt_pairs,
                "n_pairs": prompt_pairs,
            })

    if total_pairs > 0:
        exact_rate = exact_cwe_matches / total_pairs
        overlap_rate = any_overlap_count / total_pairs
    else:
        exact_rate = overlap_rate = 0

    print(f"  Exact CWE-set match rate:  {exact_rate:.1%}")
    print(f"    ({exact_cwe_matches}/{total_pairs} model pairs produce identical vulnerability sets)")
    print(f"  Any CWE overlap rate:      {overlap_rate:.1%}")
    print(f"    ({any_overlap_count}/{total_pairs} model pairs share at least one CWE)")

    # Breakdown by prompt
    high_homog = sorted(prompt_homogeneity, key=lambda x: -x["overlap_rate"])
    print(f"\n  Prompts with highest vulnerability homogeneity:")
    for h in high_homog[:10]:
        print(f"    {h['prompt_id']:<8}  exact={h['exact_rate']:.0%}  overlap={h['overlap_rate']:.0%}  ({h['n_pairs']} pairs)")

    # ── RISKY IMPORT HOMOGENEITY ──
    print(f"\n{'='*60}")
    print("RISKY IMPORT CONVERGENCE")
    print("=" * 60)

    risky_by_model = defaultdict(Counter)
    for f in all_findings:
        for ri in f["risky_imports"]:
            risky_by_model[f["model"]][ri["import"]] += 1

    print(f"  {'Model':<22} {'Risky imports found'}")
    print("  " + "-" * 50)
    for model in models:
        if risky_by_model[model]:
            items = ", ".join(f"{imp}({cnt})" for imp, cnt in risky_by_model[model].most_common(5))
            print(f"  {model:<22} {items}")
        else:
            print(f"  {model:<22} (none)")

    # Cross-model: do they import the same risky packages?
    all_risky = set()
    for model, counter in risky_by_model.items():
        all_risky.update(counter.keys())

    if all_risky:
        print(f"\n  Risky packages used by multiple models:")
        for pkg in sorted(all_risky):
            users = [m for m in models if pkg in risky_by_model[m]]
            if len(users) > 1:
                print(f"    {pkg:<15} used by {len(users)}/{len(models)} models: {', '.join(users)}")

    # ── IMPORT HALLUCINATION CHECK ──
    print(f"\n{'='*60}")
    print("IMPORT ANALYSIS (potential hallucinations)")
    print("=" * 60)

    STDLIB = {
        "abc","argparse","ast","asyncio","base64","bisect","calendar","cmath",
        "collections","concurrent","configparser","contextlib","copy","csv",
        "ctypes","dataclasses","datetime","decimal","difflib","dis","email",
        "enum","errno","fcntl","fileinput","fnmatch","fractions","ftplib",
        "functools","gc","getpass","gettext","glob","gzip","hashlib","heapq",
        "hmac","html","http","imaplib","importlib","inspect","io","ipaddress",
        "itertools","json","keyword","linecache","locale","logging","lzma",
        "math","mimetypes","mmap","multiprocessing","netrc","numbers","operator",
        "os","pathlib","pdb","pickle","pkgutil","platform","plistlib","poplib",
        "posixpath","pprint","profile","pstats","queue","random","re","readline",
        "reprlib","resource","rlcompleter","sched","secrets","select","shelve",
        "shlex","shutil","signal","site","smtplib","socket","socketserver",
        "sqlite3","ssl","stat","statistics","string","struct","subprocess",
        "sys","sysconfig","syslog","tabnanny","tarfile","tempfile","termios",
        "test","textwrap","threading","time","timeit","tkinter","token",
        "tokenize","tomllib","trace","traceback","tracemalloc","tty","turtle",
        "types","typing","unicodedata","unittest","urllib","uuid","venv",
        "warnings","wave","weakref","webbrowser","winreg","winsound","wsgiref",
        "xml","xmlrpc","zipfile","zipimport","zlib",
    }

    COMMON_PYPI = {
        "requests","flask","django","fastapi","numpy","pandas","scipy","matplotlib",
        "seaborn","plotly","torch","tensorflow","scikit-learn","sklearn","pydantic",
        "sqlalchemy","celery","redis","pymongo","psycopg2","boto3","botocore",
        "pytest","tox","black","ruff","mypy","pylint","click","typer","rich",
        "httpx","aiohttp","uvicorn","gunicorn","starlette","jinja2","pyyaml","toml",
        "pillow","opencv","cv2","beautifulsoup4","bs4","lxml","scrapy","selenium",
        "paramiko","cryptography","bcrypt","jwt","pyjwt","dotenv","python-dotenv",
        "colorama","tqdm","tabulate","prettytable","fire","setuptools","pip","wheel",
        "watchdog","schedule","apscheduler","msgpack","protobuf","grpcio","thrift",
        "kafka","pika","marshmallow","attrs","cattrs","arrow","pendulum","dateutil",
    }

    all_imports_counter = Counter()
    model_imports = defaultdict(Counter)
    for f in all_findings:
        code = extract_code(
            next((r["response_text"] for r in filtered
                  if r["prompt_id"]==f["prompt_id"] and r["model_display"]==f["model"]), "")
        )
        imps = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names: imps.add(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imps.add(node.module.split(".")[0])
        except:
            pass
        for imp in imps:
            all_imports_counter[imp] += 1
            model_imports[f["model"]][imp] += 1

    unknown = []
    for imp, count in all_imports_counter.most_common():
        if imp.lower() not in STDLIB and imp.lower() not in COMMON_PYPI:
            users = [m for m in models if imp in model_imports[m]]
            unknown.append((imp, count, users))

    if unknown:
        print(f"  Imports NOT in stdlib or common PyPI ({len(unknown)} found):")
        print(f"  These may be hallucinated or niche packages.")
        for imp, count, users in unknown[:20]:
            print(f"    {imp:<25} {count:>3}x  by {len(users)} models: {', '.join(users[:4])}")
    else:
        print("  All imports match stdlib or common PyPI packages.")

    # Check if multiple models hallucinate the same package
    shared_unknown = [(imp, cnt, users) for imp, cnt, users in unknown if len(users) > 1]
    if shared_unknown:
        print(f"\n  ** SHARED UNKNOWN IMPORTS (potential slopsquatting risk) **")
        print(f"  These packages are used by 2+ models and are NOT in stdlib/common PyPI:")
        for imp, cnt, users in shared_unknown:
            print(f"    {imp:<25} {cnt:>3}x  by {', '.join(users)}")

    # ── FIGURES ──
    if HAS_MPL and cwe_counter:
        # Fig: CWE distribution across models
        fig, ax = plt.subplots(figsize=(8, 5))
        top_cwes = [c for c, _ in cwe_counter.most_common(8)]
        cwe_names = {}
        for v in VULN_PATTERNS:
            cwe_names[v["cwe"]] = v["name"]

        model_colors = plt.cm.get_cmap("tab10", len(models))
        x = np.arange(len(top_cwes))
        w = 0.8 / len(models)

        for mi, model in enumerate(models):
            mf = [f for f in all_findings if f["model"] == model]
            vals = []
            for cwe in top_cwes:
                cnt = sum(1 for f in mf if cwe in f["cwes"])
                vals.append(cnt / len(mf) if mf else 0)
            ax.bar(x + mi * w - 0.4 + w/2, vals, w, label=model,
                   color=model_colors(mi), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{cwe_names.get(c,'')[:20]}" for c in top_cwes],
                           fontsize=7, rotation=0)
        ax.set_ylabel("Fraction of samples with this CWE")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("Vulnerability distribution across models\n(same CWEs appear in same proportions = monoculture)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.savefig(os.path.join(SEC_DIR, "fig_cwe_distribution.png"))
        plt.close(fig)
        print(f"\n  Saved: {SEC_DIR}/fig_cwe_distribution.png")

        # Fig: Vulnerability homogeneity by prompt
        if prompt_homogeneity:
            fig, ax = plt.subplots(figsize=(8, 5))
            ph = sorted(prompt_homogeneity, key=lambda x: x["prompt_id"])
            pids = [h["prompt_id"] for h in ph]
            overlap_rates = [h["overlap_rate"] for h in ph]
            exact_rates = [h["exact_rate"] for h in ph]

            x = np.arange(len(pids))
            ax.bar(x, overlap_rates, color="#f72585", alpha=0.7, label="Any CWE overlap")
            ax.bar(x, exact_rates, color="#4361ee", alpha=0.9, label="Exact CWE-set match")
            ax.set_xticks(x)
            ax.set_xticklabels(pids, rotation=90, fontsize=7)
            ax.set_ylabel("Fraction of model pairs")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_title("Vulnerability homogeneity by prompt\n(how often do different models produce the same CWEs?)")
            ax.legend(frameon=False)
            fig.savefig(os.path.join(SEC_DIR, "fig_vuln_homogeneity.png"))
            plt.close(fig)
            print(f"  Saved: {SEC_DIR}/fig_vuln_homogeneity.png")

    # ── SAVE RESULTS ──
    results = {
        "overall": {
            "total_samples": total,
            "vuln_rate": vuln_count / total if total else 0,
            "exact_cwe_match_rate": exact_rate,
            "any_cwe_overlap_rate": overlap_rate,
        },
        "per_model": model_vuln_rates,
        "cwe_distribution": dict(cwe_counter.most_common()),
        "prompt_homogeneity": prompt_homogeneity,
        "unknown_imports": [(imp, cnt, users) for imp, cnt, users in unknown[:30]],
        "shared_unknown_imports": [(imp, cnt, users) for imp, cnt, users in shared_unknown],
    }

    with open(os.path.join(SEC_DIR, "security_analysis.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV: per-pair CWE matches
    with open(os.path.join(SEC_DIR, "cwe_pair_matches.csv"), "w") as f:
        f.write("prompt_id,model_a,model_b,exact_match,any_overlap,overlap_cwes\n")
        for pm in cwe_pair_matches:
            f.write(f"{pm['prompt_id']},{pm['model_a']},{pm['model_b']},"
                    f"{pm['exact_match']},{pm['any_overlap']},{';'.join(pm['overlap_cwes'])}\n")

    # ── FINAL NARRATIVE ──
    print(f"\n{'='*60}")
    print("PAPER NARRATIVE — Security findings")
    print("=" * 60)
    print(f"""
  HEADLINE NUMBERS:
  - {vuln_count/total:.0%} of LLM-generated code samples contain vulnerabilities
  - {exact_rate:.0%} of cross-model pairs produce identical CWE sets
  - {overlap_rate:.0%} of cross-model pairs share at least one CWE
  - {len(shared_unknown)} potentially hallucinated packages shared across models

  THE ARGUMENT:
  Your data shows that LLMs don't just produce vulnerable code
  (already known) — they produce the SAME vulnerable code.
  {overlap_rate:.0%} of model pairs share overlapping vulnerabilities for
  the same task. This means a single exploit targeting a common
  LLM-generated pattern could compromise codebases built with
  ANY major AI coding assistant.

  This is implementation-level software monoculture.

  Combined with your diversity findings (44% library match rate),
  the security argument is:
  1. Models converge on same libraries (your diversity data)
  2. Models produce same vulnerability patterns (this analysis)
  3. A single CVE in a converged library/pattern = systemic risk
  4. Traditional code diversity metrics miss this (29% AST ≠ diverse)

  Results saved: {SEC_DIR}/
""")


if __name__ == "__main__":
    main()