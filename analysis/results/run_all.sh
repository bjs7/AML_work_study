#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "==> generate_bank_filter_tables.py"
python3 generate_bank_filter_tables.py

echo "==> results_analysis_comparable.py"
python3 results_analysis_comparable.py

echo "==> results_analysis_system.py"
python3 results_analysis_system.py

echo "==> results_analysis_pr_curves_comparable.py"
python3 results_analysis_pr_curves_comparable.py

echo "Done."
