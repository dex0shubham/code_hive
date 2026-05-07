"""
Compute SE diversity-security bridge analysis.
Parses inter_llm Vendi/N from summary.md, correlates with security metrics.
"""
import re, json, math
from pathlib import Path
from scipy import stats

# ── 1. Parse summary.md for inter_llm norm values ──────────────────────────
summary_path = Path(r"C:\Users\RushiHiray\Downloads\v2_se30_codet5p_centered\v2_se30_codet5p_centered\t0.7\summary.md")
text = summary_path.read_text(encoding="utf-8")

# Build {prompt: {kernel: norm_value}}
data = {}
current_prompt = None
current_kernel = None

for line in text.splitlines():
    m = re.match(r"^### (SE-\d+)", line)
    if m:
        current_prompt = m.group(1)
        data[current_prompt] = {}
        continue
    m = re.match(r"^- kernel=(\w+)", line)
    if m:
        current_kernel = m.group(1)
        continue
    if current_prompt and current_kernel and "inter_llm" in line:
        m = re.search(r"norm=([\d.]+)", line)
        if m:
            data[current_prompt][current_kernel] = float(m.group(1))

# ── 2. Security data (hardcoded from paper) ─────────────────────────────────
security = {
    "SE-01": {"match_vuln": 22.8,  "vuln_pct": 3.9},
    "SE-02": {"match_vuln": 11.3,  "vuln_pct": 18.9},
    "SE-03": {"match_vuln": 29.5,  "vuln_pct": 16.1},
    "SE-04": {"match_vuln": float("nan"), "vuln_pct": 0.0},
    "SE-05": {"match_vuln": 61.5,  "vuln_pct": 72.7},
    "SE-06": {"match_vuln": 100.0, "vuln_pct": 36.4},
    "SE-07": {"match_vuln": 80.6,  "vuln_pct": 71.1},
    "SE-08": {"match_vuln": 44.3,  "vuln_pct": 72.7},
    "SE-09": {"match_vuln": float("nan"), "vuln_pct": 0.0},
    "SE-10": {"match_vuln": 5.4,   "vuln_pct": 37.5},
    "SE-11": {"match_vuln": 76.1,  "vuln_pct": 65.9},
    "SE-12": {"match_vuln": 71.5,  "vuln_pct": 59.3},
    "SE-13": {"match_vuln": 13.0,  "vuln_pct": 31.6},
    "SE-14": {"match_vuln": 100.0, "vuln_pct": 70.7},
    "SE-15": {"match_vuln": 21.8,  "vuln_pct": 71.4},
    "SE-16": {"match_vuln": 100.0, "vuln_pct": 0.7},
    "SE-17": {"match_vuln": 78.4,  "vuln_pct": 36.1},
    "SE-18": {"match_vuln": 73.7,  "vuln_pct": 69.1},
    "SE-19": {"match_vuln": 9.2,   "vuln_pct": 5.9},
    "SE-20": {"match_vuln": 17.7,  "vuln_pct": 65.7},
    "SE-21": {"match_vuln": 78.2,  "vuln_pct": 57.3},
    "SE-22": {"match_vuln": 54.0,  "vuln_pct": 68.9},
    "SE-23": {"match_vuln": 69.6,  "vuln_pct": 6.4},
    "SE-24": {"match_vuln": 44.5,  "vuln_pct": 56.1},
    "SE-25": {"match_vuln": float("nan"), "vuln_pct": 0.2},
    "SE-26": {"match_vuln": float("nan"), "vuln_pct": 0.0},
    "SE-27": {"match_vuln": 85.6,  "vuln_pct": 61.1},
    "SE-28": {"match_vuln": 100.0, "vuln_pct": 27.0},
    "SE-29": {"match_vuln": 33.8,  "vuln_pct": 38.6},
    "SE-30": {"match_vuln": 56.5,  "vuln_pct": 64.1},
}

# ── 3. Build combined table ─────────────────────────────────────────────────
prompts = [f"SE-{i:02d}" for i in range(1, 31)]
rows = []
for p in prompts:
    row = {
        "prompt": p,
        "token_vn": data[p].get("token"),
        "ast_vn": data[p].get("ast"),
        "ct5p_vn": data[p].get("codet5p"),
        "match_vuln": security[p]["match_vuln"],
        "vuln_pct": security[p]["vuln_pct"],
    }
    rows.append(row)

# ── 4. Print table ──────────────────────────────────────────────────────────
header = f"{'Prompt':<8} {'token_vn':>9} {'ast_vn':>9} {'ct5p_vn':>9} {'match_vuln':>11} {'vuln_pct':>9}"
print(header)
print("-" * len(header))
for r in rows:
    mv = f"{r['match_vuln']:.1f}" if not math.isnan(r['match_vuln']) else "NaN"
    print(f"{r['prompt']:<8} {r['token_vn']:9.3f} {r['ast_vn']:9.3f} {r['ct5p_vn']:9.3f} {mv:>11} {r['vuln_pct']:9.1f}")

# ── 5. Grand means ──────────────────────────────────────────────────────────
token_vals = [r["token_vn"] for r in rows]
ast_vals = [r["ast_vn"] for r in rows]
ct5p_vals = [r["ct5p_vn"] for r in rows]

mean_token = sum(token_vals) / len(token_vals)
mean_ast = sum(ast_vals) / len(ast_vals)
mean_ct5p = sum(ct5p_vals) / len(ct5p_vals)

print(f"\n{'='*60}")
print("Grand mean inter-LLM Vendi/N across 30 SE prompts:")
print(f"  token:  {mean_token:.3f}   (PB ref: 0.326)")
print(f"  ast:    {mean_ast:.3f}   (PB ref: 0.121)")
print(f"  codet5p:{mean_ct5p:.3f}   (PB ref: 0.161)")

# ── 6. Spearman correlations ────────────────────────────────────────────────
# Exclude NaN match_vuln for match_vuln correlations
valid_mv = [(r["token_vn"], r["ct5p_vn"], r["match_vuln"], r["vuln_pct"])
            for r in rows if not math.isnan(r["match_vuln"])]

tok_mv = [v[0] for v in valid_mv]
ct5_mv = [v[1] for v in valid_mv]
mv_vals = [v[2] for v in valid_mv]

# For vuln_pct correlations, use all 30
tok_all = [r["token_vn"] for r in rows]
ct5_all = [r["ct5p_vn"] for r in rows]
vp_all = [r["vuln_pct"] for r in rows]

corr_results = {}

rho, p = stats.spearmanr(tok_mv, mv_vals)
corr_results["token_vn_vs_match_vuln"] = {"rho": rho, "p": p, "n": len(tok_mv)}

rho, p = stats.spearmanr(ct5_mv, mv_vals)
corr_results["ct5p_vn_vs_match_vuln"] = {"rho": rho, "p": p, "n": len(ct5_mv)}

rho, p = stats.spearmanr(tok_all, vp_all)
corr_results["token_vn_vs_vuln_pct"] = {"rho": rho, "p": p, "n": 30}

rho, p = stats.spearmanr(ct5_all, vp_all)
corr_results["ct5p_vn_vs_vuln_pct"] = {"rho": rho, "p": p, "n": 30}

print(f"\n{'='*60}")
print("Spearman rank correlations:")
for name, vals in corr_results.items():
    sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
    print(f"  {name:<30s}  rho={vals['rho']:+.3f}  p={vals['p']:.4f}  n={vals['n']}  {sig}")

# ── 7. Save JSON ────────────────────────────────────────────────────────────
out_dir = Path(r"C:\Users\RushiHiray\Documents\JDS-Main\code_hive\results")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "se_diversity_bridge.json"

# Convert NaN to None for JSON
def sanitize(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

output = {
    "grand_means": {
        "token": round(mean_token, 4),
        "ast": round(mean_ast, 4),
        "codet5p": round(mean_ct5p, 4),
    },
    "pb_reference": {"token": 0.326, "ast": 0.121, "codet5p": 0.161},
    "correlations": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                         for kk, vv in v.items()}
                     for k, v in corr_results.items()},
    "per_prompt": [{k: sanitize(v) for k, v in r.items()} for r in rows],
}

out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nSaved to {out_path}")
