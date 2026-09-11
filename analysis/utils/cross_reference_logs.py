#!/usr/bin/env python3
"""
Cross-reference parse_job_logs.py terminal output against run_overview.tex
and generate a LaTeX provenance table.

Usage:
    python cross_reference_logs.py --system system_logs.txt --comparable comparable_logs.txt
    python cross_reference_logs.py --system system_logs.txt   # comparable optional

Paste the terminal output from parse_job_logs.py (run on HPC) into text files,
then run this script locally where run_overview.tex is available.
"""
import argparse
import re
import sys
from pathlib import Path

RUN_OVERVIEW = Path(__file__).parents[3] / 'writing/Experimental-Protocol/tables/data/run_overview.tex'
OUT_TEX      = Path(__file__).parents[3] / 'writing/Experimental-Protocol/tables/data/job_provenance.tex'

# Matches a data row in run_overview.tex:
# S1 & Full-info GNN (R0, BN) & \texttt{20260306\_070318} & \texttt{20260306\_080302} \\
# Note: underscores are escaped as \_ in LaTeX
TEX_ROW_RE = re.compile(
    r'^(\w+)\s*&[^&]*&\s*(?:\\texttt\{(\d{8}\\_\d{6})\}|---)\s*&\s*(?:\\texttt\{(\d{8}\\_\d{6})\}|---)'
)

# Matches a line of parse_job_logs output with run IDs
LOG_LINE_RE = re.compile(r'^(\S+\.log)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)')


def load_run_overview(tex_path):
    """Return dict: timestamp -> (experiment_id, mode) and ordered list of (exp_id, description)."""
    ts_map   = {}  # timestamp -> (exp_id, mode)
    exp_desc = {}  # exp_id -> description (first occurrence)
    try:
        with open(tex_path) as f:
            for line in f:
                line = line.strip()
                m = TEX_ROW_RE.match(line)
                if m:
                    exp_id = m.group(1)
                    cmp_ts = m.group(2).replace('\\_', '_') if m.group(2) else None
                    sys_ts = m.group(3).replace('\\_', '_') if m.group(3) else None
                    if cmp_ts:
                        ts_map[cmp_ts] = (exp_id, 'comparable')
                    if sys_ts:
                        ts_map[sys_ts] = (exp_id, 'system')
                    # Extract description between first and second &
                    parts = line.split('&')
                    if len(parts) >= 2 and exp_id not in exp_desc:
                        exp_desc[exp_id] = parts[1].strip()
    except OSError as e:
        print(f'Error reading {tex_path}: {e}', file=sys.stderr)
        sys.exit(1)
    return ts_map, exp_desc


def parse_log_output(lines):
    """Parse lines from parse_job_logs.py terminal output."""
    rows = []
    for line in lines:
        line = line.rstrip()
        if not line or line.startswith('File') or line.startswith('---'):
            continue
        m = LOG_LINE_RE.match(line)
        if not m:
            continue
        run_ids_str = m.group(8).strip()
        run_ids = [r.strip() for r in run_ids_str.split(',') if r.strip()]
        rows.append({
            'file':      m.group(1),
            'job_id':    m.group(2),
            'partition': m.group(3),
            'node':      m.group(4),
            'gpus':      m.group(5),
            'elapsed':   m.group(6),
            'run_ids':   run_ids,
        })
    return rows


def build_exp_job_map(rows, ts_map):
    """
    Build: exp_id -> mode -> {job_id, partition, node, gpus, elapsed}
    One job can produce multiple experiments (grouped jobs).
    """
    exp_map = {}
    for r in rows:
        for ts in r['run_ids']:
            if ts not in ts_map:
                continue
            exp_id, mode = ts_map[ts]
            if exp_id not in exp_map:
                exp_map[exp_id] = {}
            exp_map[exp_id][mode] = {
                'job_id':    r['job_id'],
                'partition': r['partition'],
                'node':      r['node'],
                'gpus':      r['gpus'],
                'elapsed':   r['elapsed'],
            }
    return exp_map


def gpu_label(partition):
    if 'h100' in partition:
        return 'H100'
    if 'a100' in partition:
        return 'A100'
    if 'v100' in partition:
        return 'V100'
    return partition


def print_cross_reference(rows, ts_map):
    col = 45
    all_log_ts = set()
    for r in rows:
        matches = []
        for ts in r['run_ids']:
            all_log_ts.add(ts)
            if ts in ts_map:
                exp_id, mode = ts_map[ts]
                matches.append(f'{exp_id} ({mode}) [{ts}]')
            else:
                matches.append(f'NOT IN TABLE [{ts}]')
        match_str = ', '.join(matches) if matches else '(no run IDs in log)'
        print(f"{r['file']:<{col}} job={r['job_id']}  {r['partition']}  elapsed={r['elapsed']}")
        print(f"  {'':>{col-2}} → {match_str}")
        print()
    return all_log_ts


def write_provenance_tex(exp_map, exp_desc, ts_map, out_path):
    """Write a LaTeX provenance table covering both modes."""

    # Collect ordered exp_ids from run_overview (preserves table order)
    seen = set()
    ordered = []
    for ts, (exp_id, _) in ts_map.items():
        if exp_id not in seen and exp_id in exp_map:
            seen.add(exp_id)
            ordered.append(exp_id)

    lines = []
    lines.append(r'% Job provenance table — auto-generated by cross_reference_logs.py')
    lines.append(r'% Columns: experiment, GPU type, SLURM job ID, node, elapsed (per mode)')
    lines.append(r'\begin{tabular}{llllllll}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{ID} & \textbf{Description} & '
        r'\multicolumn{3}{c}{\textbf{Comparable}} & '
        r'\multicolumn{3}{c}{\textbf{System}} \\'
    )
    lines.append(
        r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}'
    )
    lines.append(
        r' & & \textbf{GPU} & \textbf{Job ID} & \textbf{Elapsed} '
        r'& \textbf{GPU} & \textbf{Job ID} & \textbf{Elapsed} \\'
    )
    lines.append(r'\midrule')

    for exp_id in ordered:
        desc = exp_desc.get(exp_id, '')
        cmp  = exp_map[exp_id].get('comparable', {})
        sys  = exp_map[exp_id].get('system', {})

        cmp_gpu     = gpu_label(cmp.get('partition', ''))
        cmp_job     = cmp.get('job_id', '---').replace('_', '\\_')
        cmp_elapsed = cmp.get('elapsed', '---').replace('_', '\\_')

        sys_gpu     = gpu_label(sys.get('partition', ''))
        sys_job     = sys.get('job_id', '---').replace('_', '\\_')
        sys_elapsed = sys.get('elapsed', '---').replace('_', '\\_')

        lines.append(
            f'{exp_id} & {desc} & '
            f'{cmp_gpu} & \\texttt{{{cmp_job}}} & {cmp_elapsed} & '
            f'{sys_gpu} & \\texttt{{{sys_job}}} & {sys_elapsed} \\\\'
        )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nProvenance table written to: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Cross-reference SLURM logs against run_overview.tex.')
    parser.add_argument('--system',     metavar='FILE', help='parse_job_logs output for system mode')
    parser.add_argument('--comparable', metavar='FILE', help='parse_job_logs output for comparable mode')
    parser.add_argument('--tex', metavar='FILE', help=f'Output .tex path (default: {OUT_TEX})')
    args = parser.parse_args()

    if not args.system and not args.comparable:
        print(__doc__)
        sys.exit(1)

    ts_map, exp_desc = load_run_overview(RUN_OVERVIEW)
    print(f'Loaded {len(ts_map)} timestamps from run_overview.tex\n')

    sys_rows = []
    cmp_rows = []

    if args.system:
        with open(args.system) as f:
            sys_rows = parse_log_output(f.readlines())
        print(f'=== SYSTEM ({len(sys_rows)} log files) ===')
        print_cross_reference(sys_rows, ts_map)

    if args.comparable:
        with open(args.comparable) as f:
            cmp_rows = parse_log_output(f.readlines())
        print(f'=== COMPARABLE ({len(cmp_rows)} log files) ===')
        print_cross_reference(cmp_rows, ts_map)

    # Report timestamps in run_overview.tex not seen in any log
    all_log_ts = {ts for r in sys_rows + cmp_rows for ts in r['run_ids']}
    unmatched  = {ts: info for ts, info in ts_map.items() if ts not in all_log_ts}
    if unmatched:
        print('=' * 90)
        print('Timestamps in run_overview.tex NOT found in any log file:')
        print('=' * 90)
        for ts, (exp_id, mode) in sorted(unmatched.items()):
            print(f'  {exp_id:<5} ({mode:<10}) {ts}')
    else:
        print('All run_overview.tex timestamps accounted for.')

    # Build provenance map and write .tex
    exp_map = {}
    for rows in [sys_rows, cmp_rows]:
        for exp_id, mode_data in build_exp_job_map(rows, ts_map).items():
            if exp_id not in exp_map:
                exp_map[exp_id] = {}
            exp_map[exp_id].update(mode_data)

    out_path = Path(args.tex) if args.tex else OUT_TEX
    write_provenance_tex(exp_map, exp_desc, ts_map, out_path)


if __name__ == '__main__':
    main()
