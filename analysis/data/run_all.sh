#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "==> data_analysis.py"
python3 data_analysis.py

echo "==> data_summary_table.py"
python3 data_summary_table.py

echo "==> data_split_bank_partici_tables.py"
python3 data_split_bank_partici_tables.py

echo "==> bank_analysis_comparable.py"
python3 bank_analysis_comparable.py

echo "==> bank_analysis_system.py"
python3 bank_analysis_system.py

echo "==> inter_intra_full_data.py"
python3 inter_intra_full_data.py

echo "Done."
