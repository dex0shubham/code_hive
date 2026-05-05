#!/usr/bin/env python3
"""
pypi_check.py — Live PyPI existence and metadata lookup for slopsquatting analysis.

Replaces the static `STDLIB` + `COMMON_PYPI` allowlist in security_analysis.py
with a live PyPI JSON API check. Implements the methodology from Spracklen et
al. (USENIX Security 2025) for measuring package hallucinations:

  1. exists?            HTTP GET https://pypi.org/pypi/<name>/json -> 200 vs 404
  2. metadata           creation date, maintainer, download count
  3. recurrence         given a list of (prompt_id, sample_idx, hallucinated_name),
                        fraction of hallucinations that recur on the same prompt
                        across samples (the "deterministic" fraction)
  4. cross-ecosystem    flag Python imports whose names are valid npm packages

Cache: writes results to <ROOT>/results/pypi_cache.jsonl. Re-uses cached
answers for ~7 days; a name is rechecked if older than that or if --force.

Usage as a module:
    from pypi_check import package_exists, package_metadata, recurrence_rate

Usage as a CLI:
    python pypi_check.py --check django requests fakepkg-12345
    python pypi_check.py --recurrence-from results/proof_security/.../slop.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

try:
    import httpx  # already in requirements
    _CLIENT = httpx.Client(timeout=10.0)
except ImportError:
    import urllib.request, urllib.error  # std-lib fallback
    _CLIENT = None

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "results" / "pypi_cache.jsonl"
NPM_CACHE_PATH = ROOT / "results" / "npm_cache.jsonl"

_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


# ─────────────────────────────────────────────────────────────────────
# Cache layer (append-only JSONL, last entry wins per-key)
# ─────────────────────────────────────────────────────────────────────

_pypi_cache: dict[str, dict] | None = None
_npm_cache: dict[str, dict] | None = None


def _load_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["name"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _save_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _is_fresh(rec: dict) -> bool:
    return (time.time() - rec.get("fetched_at_unix", 0)) < _TTL_SECONDS


# ─────────────────────────────────────────────────────────────────────
# Live PyPI check
# ─────────────────────────────────────────────────────────────────────

def _http_get_json(url: str) -> tuple[int, Optional[dict]]:
    """Return (status_code, parsed_json_or_None). Never raises."""
    try:
        if _CLIENT is not None:
            r = _CLIENT.get(url)
            return r.status_code, (r.json() if r.status_code == 200 else None)
        else:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return resp.status, data
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return -1, None


def package_exists(name: str, force: bool = False) -> bool:
    """True iff `name` resolves on PyPI (status 200)."""
    rec = package_metadata(name, force=force)
    return bool(rec.get("exists"))


def package_metadata(name: str, force: bool = False) -> dict:
    """Cached PyPI metadata for `name`. Always returns a dict with at least
    {'name', 'exists', 'fetched_at_unix'}; if exists, also adds first-release
    date, latest-release date, author, project_url, home_page, summary."""
    global _pypi_cache
    if _pypi_cache is None:
        _pypi_cache = _load_cache(CACHE_PATH)

    cached = _pypi_cache.get(name)
    if cached and _is_fresh(cached) and not force:
        return cached

    code, data = _http_get_json(f"https://pypi.org/pypi/{name}/json")
    rec: dict = {
        "name": name,
        "exists": (code == 200 and data is not None),
        "status_code": code,
        "fetched_at_unix": time.time(),
    }
    if rec["exists"]:
        info = data.get("info", {}) if data else {}
        releases = data.get("releases", {}) if data else {}
        # First and latest upload dates
        all_dates = []
        for ver_files in releases.values():
            for f in (ver_files or []):
                upload = f.get("upload_time_iso_8601") or f.get("upload_time")
                if upload:
                    all_dates.append(upload)
        all_dates.sort()
        rec.update({
            "summary": (info.get("summary") or "")[:200],
            "author": info.get("author") or info.get("maintainer"),
            "author_email": info.get("author_email") or info.get("maintainer_email"),
            "home_page": info.get("home_page") or info.get("project_url") or "",
            "first_upload_iso": all_dates[0] if all_dates else None,
            "latest_upload_iso": all_dates[-1] if all_dates else None,
            "n_releases": len(releases),
            "yanked": bool(info.get("yanked")) or False,
        })

    _save_cache(CACHE_PATH, rec)
    _pypi_cache[name] = rec
    return rec


# ─────────────────────────────────────────────────────────────────────
# Cross-ecosystem confusion (Spracklen finding: 8.7% of hallucinated
# Python packages are valid npm packages).
# ─────────────────────────────────────────────────────────────────────

def package_exists_on_npm(name: str, force: bool = False) -> bool:
    """True iff `name` resolves on the npm registry."""
    global _npm_cache
    if _npm_cache is None:
        _npm_cache = _load_cache(NPM_CACHE_PATH)
    cached = _npm_cache.get(name)
    if cached and _is_fresh(cached) and not force:
        return bool(cached.get("exists"))

    code, _ = _http_get_json(f"https://registry.npmjs.org/{name}")
    rec = {
        "name": name,
        "exists": (code == 200),
        "status_code": code,
        "fetched_at_unix": time.time(),
    }
    _save_cache(NPM_CACHE_PATH, rec)
    _npm_cache[name] = rec
    return rec["exists"]


# ─────────────────────────────────────────────────────────────────────
# Suspicious-package flags (squat signature)
# ─────────────────────────────────────────────────────────────────────

def is_suspicious_recent(meta: dict, threshold_days: int = 30) -> bool:
    """A real package whose first upload is < threshold_days old: candidate
    squat (someone may have registered it after seeing LLM hallucinations)."""
    if not meta.get("exists"):
        return False
    first = meta.get("first_upload_iso")
    if not first:
        return False
    try:
        # Parse ISO 8601 (drop tz suffix variants)
        from datetime import datetime, timezone
        ts = first.replace("Z", "+00:00")
        if "." in ts:  # truncate microseconds beyond 6 digits
            head, frac = ts.split(".", 1)
            tz = ""
            for sep in ("+", "-"):
                if sep in frac:
                    frac, _, tz_part = frac.partition(sep)
                    tz = sep + tz_part
                    break
            ts = f"{head}.{frac[:6]}{tz}"
        dt = datetime.fromisoformat(ts)
        age_days = (datetime.now(timezone.utc) - dt).days
        return age_days < threshold_days
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Recurrence-rate analysis (Spracklen 43%)
# ─────────────────────────────────────────────────────────────────────

def recurrence_rate(hallucinations: list[dict]) -> dict:
    """Given list of {prompt_id, sample_idx, package} dicts, compute:

    - **deterministic_rate** — fraction of (prompt_id, package) pairs that
      appear in *every* sample of that prompt
    - **stochastic_rate**    — fraction that appear in 2+ but not all samples
    - **once_only_rate**     — fraction that appear in exactly one sample
    - per-prompt breakdown
    """
    by_prompt: dict[str, list[str]] = defaultdict(list)
    samples_per_prompt: dict[str, set] = defaultdict(set)
    for h in hallucinations:
        by_prompt[h["prompt_id"]].append(h["package"])
        samples_per_prompt[h["prompt_id"]].add(h.get("sample_idx", -1))

    determ = 0
    stoch = 0
    once = 0
    total_unique_pairs = 0
    per_prompt_breakdown: dict[str, dict] = {}

    # Need to know how many *samples* exist per prompt overall, not just
    # samples that produced hallucinations. The caller can pre-compute this
    # and pass via `n_samples_by_prompt`; otherwise we approximate.
    for pid, pkgs in by_prompt.items():
        counts = Counter(pkgs)
        # Approximate: max count in this prompt is the per-prompt sample budget
        n_samples = max(samples_per_prompt[pid].__len__(), 1)
        determ_p, stoch_p, once_p = 0, 0, 0
        for pkg, c in counts.items():
            total_unique_pairs += 1
            if c >= n_samples:
                determ += 1; determ_p += 1
            elif c > 1:
                stoch += 1; stoch_p += 1
            else:
                once += 1; once_p += 1
        per_prompt_breakdown[pid] = {
            "n_unique_pkgs": len(counts),
            "deterministic": determ_p,
            "stochastic": stoch_p,
            "once_only": once_p,
        }

    if total_unique_pairs == 0:
        return {"deterministic_rate": 0.0, "stochastic_rate": 0.0,
                "once_only_rate": 0.0, "n_unique_pairs": 0,
                "per_prompt": {}}

    return {
        "deterministic_rate": determ / total_unique_pairs,
        "stochastic_rate":    stoch  / total_unique_pairs,
        "once_only_rate":     once   / total_unique_pairs,
        "n_unique_pairs":     total_unique_pairs,
        "per_prompt":         per_prompt_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=False)

    p_check = sub.add_parser("check", help="Check whether package(s) exist on PyPI")
    p_check.add_argument("names", nargs="+")
    p_check.add_argument("--force", action="store_true",
                         help="Bypass cache and re-fetch.")

    p_npm = sub.add_parser("npm-check", help="Check whether package(s) exist on npm")
    p_npm.add_argument("names", nargs="+")

    p_rec = sub.add_parser("recurrence", help="Compute deterministic/stochastic rates from a JSONL of hallucinations")
    p_rec.add_argument("path")

    args = p.parse_args()

    if args.cmd == "check":
        for n in args.names:
            meta = package_metadata(n, force=args.force)
            mark = "OK " if meta["exists"] else "404"
            extra = ""
            if meta["exists"]:
                extra = (f" first={meta.get('first_upload_iso','?')[:10]}  "
                         f"author={(meta.get('author') or '')[:30]:30s}  "
                         f"yanked={meta.get('yanked', False)}")
                if is_suspicious_recent(meta):
                    extra += "  [SUSPICIOUS: <30 days old]"
            print(f"  {mark}  {n:30s}{extra}")
    elif args.cmd == "npm-check":
        for n in args.names:
            print(f"  {'OK ' if package_exists_on_npm(n) else '404'}  npm:{n}")
    elif args.cmd == "recurrence":
        rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
        result = recurrence_rate(rows)
        print(json.dumps(result, indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    _cli()
