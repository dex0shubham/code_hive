#!/usr/bin/env bash
set -euo pipefail
SCAN_RESULTS=${1:?Usage: ./run_recommended_design.sh CALIBRATED_SCAN_ZIP_OR_JSONL [OUT_DIR]}
OUT_DIR=${2:-manual_verification_out}
EXEC_FLAG=${EXEC_FLAG---execute}
TIMEOUT=${TIMEOUT:-4}
mkdir -p "$OUT_DIR"
python3 verify_unsafe_deserialization.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 50 $EXEC_FLAG --timeout "$TIMEOUT"
python3 verify_unsafe_eval.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 60 $EXEC_FLAG --timeout "$TIMEOUT"
python3 verify_command_injection.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 80 $EXEC_FLAG --timeout "$TIMEOUT"
python3 audit_path_traversal.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 60 $EXEC_FLAG --timeout "$TIMEOUT"
python3 audit_mass_assignment.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 60 $EXEC_FLAG --timeout "$TIMEOUT"
python3 audit_sql_injection.py --scan-results "$SCAN_RESULTS" --out-dir "$OUT_DIR" --limit 0 --timeout "$TIMEOUT"
python3 report_manual_stats.py --inputs "$OUT_DIR" --out-dir "$OUT_DIR/report"
