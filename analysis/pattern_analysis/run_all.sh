#!/usr/bin/env bash
# Run all pattern_analysis scripts in dependency order.
# Scripts that share the same data split are grouped; fragmented before
# nonfragmented, comparable before system, structure last.
set -e
cd "$(dirname "$0")"

echo "==> pattern_analysis_fragmented_comparable.py"
python3 pattern_analysis_fragmented_comparable.py

echo "==> pattern_analysis_nonfragmented_comparable.py"
python3 pattern_analysis_nonfragmented_comparable.py

echo "==> results_analysis_pattern_structure_comparable.py"
python3 results_analysis_pattern_structure_comparable.py

echo "==> pattern_analysis_fragmented_system.py"
python3 pattern_analysis_fragmented_system.py

echo "==> pattern_analysis_nonfragmented_system.py"
python3 pattern_analysis_nonfragmented_system.py

echo "==> results_analysis_pattern_structure_system.py"
python3 results_analysis_pattern_structure_system.py

echo "==> a_pattern_descrip_percentage_attempt_fragmen.py"
python3 a_pattern_descrip_percentage_attempt_fragmen.py

echo "Done."
