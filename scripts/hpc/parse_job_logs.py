#!/usr/bin/env python3
"""
Parse SLURM log files to extract hardware info, elapsed time, and experiment run IDs.

Tries sacct first for elapsed/state; falls back to parsing first/last timestamps
from the log itself when sacct records are no longer available (jobs too old).
Also extracts experiment directory timestamps from "Results saved" lines so
they can be matched against run_overview.tex.

Run this on the HPC.

Usage:
    python parse_job_logs.py <log_dir>
    python parse_job_logs.py <log_dir> --csv job_summary.csv

Examples:
    python parse_job_logs.py /data/leuven/362/vsc36278/AML_work_study/batch_jobs/logs/logs_for_current_results
    python parse_job_logs.py . --csv summary.csv
"""
import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Matches log lines like: 2026-03-06 17:18:42 [INFO ] ...
LOG_TIMESTAMP_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

# Matches "Results saved successfully to: .../20260306_171841"
RESULTS_SAVED_RE = re.compile(r'Results saved successfully to:\s+(\S+)')

# Matches the experiment timestamp at the end of the path (YYYYMMDD_HHMMSS)
RUN_ID_RE = re.compile(r'(\d{8}_\d{6})$')


def parse_log_header(log_path):
    """Extract SLURM variables from the log file header."""
    info = {}
    keys = {
        'SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_NODELIST',
        'SLURM_JOB_GPUS', 'SLURM_JOB_CPUS_PER_NODE', 'SLURM_CLUSTER_NAME',
        'SLURM_JOB_NAME', 'Walltime',
    }
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                for key in keys:
                    if line.startswith(key + ':'):
                        info[key] = line.split(':', 1)[1].strip()
                if '=======' in line and len(info) > 3:
                    break
    except OSError:
        pass
    return info


def get_sacct_info(job_id, cluster=None):
    """Query sacct for elapsed/start/end/state. Returns empty dict if unavailable."""
    cmd = [
        'sacct', '-j', job_id,
        '--format=Elapsed,Start,End,State',
        '--noheader', '-P',
    ]
    if cluster:
        cmd += ['-M', cluster]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = [
            l for l in result.stdout.strip().split('\n')
            if l and '.batch' not in l and '.extern' not in l
        ]
        if lines:
            parts = lines[0].split('|')
            if len(parts) >= 4 and parts[0] not in ('', 'Unknown'):
                return {
                    'elapsed': parts[0],
                    'start':   parts[1],
                    'end':     parts[2],
                    'state':   parts[3],
                    'source':  'sacct',
                }
    except Exception:
        pass
    return {}


def parse_log_timestamps(log_path):
    """
    Fall back: scan the log for Python logger timestamps to get first/last time
    and extract experiment run IDs from 'Results saved' lines.
    """
    first_ts = None
    last_ts  = None
    run_ids  = []

    try:
        with open(log_path) as f:
            for line in f:
                m = LOG_TIMESTAMP_RE.match(line)
                if m:
                    ts_str = m.group(1)
                    if first_ts is None:
                        first_ts = ts_str
                    last_ts = ts_str

                m2 = RESULTS_SAVED_RE.search(line)
                if m2:
                    path = m2.group(1)
                    m3 = RUN_ID_RE.search(path)
                    if m3:
                        run_ids.append(m3.group(1))
    except OSError:
        pass

    elapsed = None
    if first_ts and last_ts and first_ts != last_ts:
        fmt = '%Y-%m-%d %H:%M:%S'
        try:
            delta = datetime.strptime(last_ts, fmt) - datetime.strptime(first_ts, fmt)
            total = int(delta.total_seconds())
            h, rem = divmod(total, 3600)
            m, s   = divmod(rem, 60)
            elapsed = f'{h:02d}:{m:02d}:{s:02d}'
        except ValueError:
            pass

    return {
        'elapsed': elapsed or '?',
        'start':   first_ts or '?',
        'end':     last_ts  or '?',
        'state':   '?',
        'source':  'log_timestamps',
        'run_ids': run_ids,
    }


def main():
    parser = argparse.ArgumentParser(description='Summarise SLURM log files.')
    parser.add_argument('log_dir', help='Directory containing .log files')
    parser.add_argument('--csv', metavar='FILE', help='Also write results to a CSV file')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob('*.log'))

    if not logs:
        print(f'No .log files found in {log_dir}')
        sys.exit(1)

    rows = []
    for log_path in logs:
        match = re.search(r'_(\d{6,})\.log$', log_path.name)
        if not match:
            continue
        job_id = match.group(1)

        header  = parse_log_header(log_path)
        cluster = header.get('SLURM_CLUSTER_NAME', 'wice')

        timing = get_sacct_info(job_id, cluster)
        if not timing:
            timing = parse_log_timestamps(log_path)
        else:
            # Still parse run_ids from the log even when sacct works
            log_info = parse_log_timestamps(log_path)
            timing['run_ids'] = log_info.get('run_ids', [])

        rows.append({
            'file':      log_path.name,
            'job_id':    job_id,
            'job_name':  header.get('SLURM_JOB_NAME', '?'),
            'cluster':   cluster,
            'partition': header.get('SLURM_JOB_PARTITION', '?'),
            'node':      header.get('SLURM_NODELIST', '?'),
            'gpus':      header.get('SLURM_JOB_GPUS', '?'),
            'cpus':      header.get('SLURM_JOB_CPUS_PER_NODE', '?'),
            'walltime':  header.get('Walltime', '?'),
            'elapsed':   timing.get('elapsed', '?'),
            'start':     timing.get('start', '?'),
            'end':       timing.get('end', '?'),
            'state':     timing.get('state', '?'),
            'source':    timing.get('source', '?'),
            'run_ids':   ', '.join(timing.get('run_ids', [])),
        })

    # Print table
    col_file = 45
    col_jid  = 10
    col_part = 13
    col_node = 12
    col_gpu  = 6
    col_ela  = 10
    col_sta  = 8

    hdr = (f"{'File':<{col_file}} {'Job ID':<{col_jid}} {'Partition':<{col_part}} {'Node':<{col_node}} "
           f"{'GPUs':<{col_gpu}} {'Elapsed':<{col_ela}} {'Source':<{col_sta}}  Run IDs (from saved paths)")
    print(hdr)
    print('-' * (len(hdr) + 20))
    for r in rows:
        print(
            f"{r['file']:<{col_file}} {r['job_id']:<{col_jid}} {r['partition']:<{col_part}} {r['node']:<{col_node}} "
            f"{r['gpus']:<{col_gpu}} {r['elapsed']:<{col_ela}} {r['source']:<{col_sta}}  {r['run_ids']}"
        )

    if args.csv:
        csv_path = Path(args.csv)
        fields = ['file', 'job_id', 'job_name', 'cluster', 'partition', 'node',
                  'gpus', 'cpus', 'walltime', 'elapsed', 'start', 'end', 'state', 'source', 'run_ids']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nCSV written to: {csv_path}')


if __name__ == '__main__':
    main()
