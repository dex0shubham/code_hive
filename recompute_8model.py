"""
Recompute all security-pillar statistics for the 8-model producing subset.
Reads scan_results.jsonl (11-model, 13,199 samples), filters to 8 producing models,
and outputs the updated numbers for paper_final.tex.
"""
import json
import math
from collections import Counter, defaultdict

SCAN_FILE = r"C:\Users\RushiHiray\Documents\JDS-Main\code_hive\results\proof_security\sec_calibrated_v2\scan_results.jsonl"
CALIB_FILE = r"C:\Users\RushiHiray\Documents\JDS-Main\code_hive\results\proof_security\sec_calibrated_v2\calibration_audit.json"

EXCLUDED_MODELS = {"Claude 3.5 Haiku", "Gemini 2.5 Flash", "Gemini 2.5 Pro"}

# Load all samples
all_samples = []
with open(SCAN_FILE) as f:
    for line in f:
        all_samples.append(json.loads(line))

print(f"Total samples (11-model): {len(all_samples)}")

# Filter to 8 producing models
samples = [s for s in all_samples if s["model_display"] not in EXCLUDED_MODELS]
print(f"Filtered samples (8-model): {len(samples)}")

# Check model list
models_8 = sorted(set(s["model_display"] for s in samples))
print(f"Models: {models_8}")

# ============================================================
# PILLAR S1: Calibrated vulnerability rate
# ============================================================
n_total = len(samples)
n_vuln = sum(1 for s in samples if s["is_vulnerable"])
vuln_rate = n_vuln / n_total
# Wilson CI
z = 1.96
p_hat = vuln_rate
ci_lo = (p_hat + z*z/(2*n_total) - z*math.sqrt(p_hat*(1-p_hat)/n_total + z*z/(4*n_total*n_total))) / (1 + z*z/n_total)
ci_hi = (p_hat + z*z/(2*n_total) + z*math.sqrt(p_hat*(1-p_hat)/n_total + z*z/(4*n_total*n_total))) / (1 + z*z/n_total)

print(f"\n=== PILLAR S1: Calibrated vulnerability rate ===")
print(f"Overall vuln rate: {vuln_rate:.1%} (CI {ci_lo:.1%}--{ci_hi:.1%})")

# Severity
sev_vals = [s["severity_total"] for s in samples if s["is_vulnerable"]]
if sev_vals:
    sev_mean = sum(sev_vals) / len(sev_vals)
    print(f"Mean severity (vuln samples): {sev_mean:.2f}")

# CWE distribution
cwe_counter = Counter()
for s in samples:
    for cwe in s["cwe_set"]:
        cwe_counter[cwe] += 1

print(f"\nTop CWEs after calibration:")
for cwe, count in cwe_counter.most_common(10):
    print(f"  {cwe}: {count}")

# Total findings (sum of all CWE hits across all samples)
total_findings = sum(cwe_counter.values())
print(f"\nTotal CWE findings: {total_findings}")

# Per-model vulnerability rates
print(f"\n=== PER-MODEL VULNERABILITY RATES ===")
model_stats = defaultdict(lambda: {"n": 0, "vuln": 0})
for s in samples:
    m = s["model_display"]
    model_stats[m]["n"] += 1
    if s["is_vulnerable"]:
        model_stats[m]["vuln"] += 1

for m in sorted(model_stats.keys(), key=lambda x: model_stats[x]["vuln"]/model_stats[x]["n"]):
    ms = model_stats[m]
    rate = ms["vuln"] / ms["n"]
    print(f"  {m:25s}: {rate:.1%} ({ms['vuln']}/{ms['n']})")

# ============================================================
# CALIBRATION RULE FIRES — recompute per model subset
# ============================================================
# We need to recount rule fires from the raw data
# Since we have the calibrated scan_results, we need to check
# which rule fires came from which models.
# Actually, the rule fires are baked into scan_results.jsonl already.
# The calibration_audit.json counts are for all 11 models.
# We need to estimate what the 8-model rule fire counts would be.
# But actually: the scan_results already has the POST-calibration findings.
# Let's count the actual CWE distribution for the 8-model subset.

print(f"\n=== CALIBRATION FIRE ESTIMATES ===")
print("(Rule fires need to be recomputed from raw pre-calibration data)")
print("The CWE counts above reflect post-calibration 8-model subset.")

# The total calibrated findings
total_calib_findings = sum(len(s["findings"]) for s in samples)
print(f"Total calibrated findings (8-model): {total_calib_findings}")

# ============================================================
# PILLAR S2: Cross-model CWE-pattern homogeneity
# ============================================================
print(f"\n=== PILLAR S2: Cross-model pattern homogeneity ===")

# For each prompt, compute cross-model exact-match rate on pattern signatures
# restricted to vulnerable samples only
prompt_homogeneity = {}
for prompt_id in sorted(set(s["prompt_id"] for s in samples)):
    prompt_samples = [s for s in samples if s["prompt_id"] == prompt_id and s["is_vulnerable"]]
    if len(prompt_samples) < 2:
        prompt_homogeneity[prompt_id] = None
        continue

    # Get pattern signatures (as frozensets for comparison)
    sigs = [tuple(sorted(s["pattern_signature"])) if s["pattern_signature"] else () for s in prompt_samples]

    # Compute pairwise exact-match rate
    n_pairs = 0
    n_match = 0
    for i in range(len(sigs)):
        for j in range(i+1, len(sigs)):
            n_pairs += 1
            if sigs[i] == sigs[j] and sigs[i] != ():
                n_match += 1

    match_rate = (n_match / n_pairs * 100) if n_pairs > 0 else 0
    vuln_pct = len(prompt_samples) / len([s for s in samples if s["prompt_id"] == prompt_id]) * 100
    prompt_homogeneity[prompt_id] = {
        "match_vuln": round(match_rate, 1),
        "vuln_pct": round(vuln_pct, 1),
        "n_vuln": len(prompt_samples),
        "n_total": len([s for s in samples if s["prompt_id"] == prompt_id])
    }

print(f"\nPer-prompt homogeneity (8-model subset):")
print(f"{'Prompt':8s} {'match_vuln':>10s} {'vuln_pct':>8s} {'n_vuln':>6s}")
for pid in sorted(prompt_homogeneity.keys()):
    v = prompt_homogeneity[pid]
    if v is None:
        print(f"{pid:8s} {'N/A':>10s} {'0.0':>8s}")
    else:
        print(f"{pid:8s} {v['match_vuln']:>10.1f} {v['vuln_pct']:>8.1f} {v['n_vuln']:>6d}")

# Count prompts with 100% match and >= 78%
prompts_100 = [p for p, v in prompt_homogeneity.items() if v and v["match_vuln"] == 100.0]
prompts_78 = [p for p, v in prompt_homogeneity.items() if v and v["match_vuln"] >= 78.0]
print(f"\nPrompts with 100% match: {len(prompts_100)} ({prompts_100})")
print(f"Prompts with >= 78% match: {len(prompts_78)} ({prompts_78})")

# ============================================================
# SE BRIDGE: Diversity metrics comparison
# ============================================================
# The se_diversity_bridge.json has per-prompt Vendi/N values
# These were computed from v2_se30 data which used N=44 (all 11 models)
# We need to note this discrepancy
print(f"\n=== SE BRIDGE NOTE ===")
print("The SE diversity bridge Vendi/N values in se_diversity_bridge.json")
print("were computed with N=44 (all 11 models, 4 samples each).")
print("These need recomputation with N=32 (8 models, 4 samples each).")
print("This requires re-running pipeline.py with filtered raw data.")

# ============================================================
# Correlation recomputation
# ============================================================
# We can recompute correlations using the 8-model match_vuln and the existing Vendi/N
# But the Vendi/N values are from the 11-model run, so the correlation is approximate
# Let's compute with the new match_vuln values
print(f"\n=== SPEARMAN CORRELATION (8-model match_vuln) ===")

# Get per-prompt Vendi/N from existing bridge data
bridge = json.load(open(r"C:\Users\RushiHiray\Documents\JDS-Main\code_hive\results\se_diversity_bridge.json"))
per_prompt_bridge = {p["prompt"]: p for p in bridge["per_prompt"]}

# Pair with new match_vuln
pairs_ct5p = []
pairs_token = []
for pid, hom in prompt_homogeneity.items():
    if hom is None or hom["match_vuln"] is None:
        continue
    bp = per_prompt_bridge.get(pid)
    if bp is None:
        continue
    pairs_ct5p.append((bp["ct5p_vn"], hom["match_vuln"]))
    pairs_token.append((bp["token_vn"], hom["match_vuln"]))

# Simple Spearman using ranks
def spearman(pairs):
    n = len(pairs)
    if n < 3:
        return None, None, n
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = rank(xs)
    ry = rank(ys)

    d_sq = sum((rx[i] - ry[i])**2 for i in range(n))
    rho = 1 - 6 * d_sq / (n * (n*n - 1))

    # Approximate p-value using t-distribution approximation
    if abs(rho) < 1:
        t_stat = rho * math.sqrt((n-2)/(1-rho*rho))
        # Simple two-tailed p from t approximation
        # Using a rough p-value estimation
        df = n - 2
        p_approx = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))  # normal approx
    else:
        p_approx = 0.0

    return rho, p_approx, n

rho_ct5p, p_ct5p, n_ct5p = spearman(pairs_ct5p)
rho_token, p_token, n_token = spearman(pairs_token)

print(f"CodeT5+ Vendi/N vs 8-model match_vuln: rho={rho_ct5p:.4f}, p~{p_ct5p:.4f}, n={n_ct5p}")
print(f"Token Vendi/N vs 8-model match_vuln:    rho={rho_token:.4f}, p~{p_token:.4f}, n={n_token}")
print()
print("NOTE: Vendi/N values are still from the 11-model pool (N=44).")
print("For full accuracy, re-run pipeline.py on 8-model filtered SE data.")

# ============================================================
# Output summary JSON
# ============================================================
output = {
    "n_models": 8,
    "models": models_8,
    "n_samples": n_total,
    "pillar_s1": {
        "vuln_rate": round(vuln_rate, 4),
        "vuln_rate_pct": round(vuln_rate * 100, 1),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "ci95_pct": [round(ci_lo*100, 1), round(ci_hi*100, 1)],
        "n_vulnerable": n_vuln,
        "severity_mean_vuln": round(sev_mean, 2) if sev_vals else None,
        "total_cwe_findings": total_findings,
        "cwe_distribution": dict(cwe_counter.most_common()),
    },
    "per_model": {m: {"n": model_stats[m]["n"], "vuln_rate": round(model_stats[m]["vuln"]/model_stats[m]["n"], 4)} for m in models_8},
    "pillar_s2": {
        "per_prompt": prompt_homogeneity,
        "prompts_100pct": prompts_100,
        "prompts_ge78pct": prompts_78,
    },
    "correlation_approx": {
        "ct5p_vs_match_vuln": {"rho": round(rho_ct5p, 4) if rho_ct5p else None, "p": round(p_ct5p, 4) if p_ct5p else None, "n": n_ct5p},
        "token_vs_match_vuln": {"rho": round(rho_token, 4) if rho_token else None, "p": round(p_token, 4) if p_token else None, "n": n_token},
        "note": "Vendi/N values from 11-model pool; match_vuln from 8-model subset"
    }
}

out_path = r"C:\Users\RushiHiray\Documents\JDS-Main\code_hive\results\recomputed_8model.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
