#!/usr/bin/env python3
# Format sm-list output: reads tab-separated CreationTime/TrainingEndTime/Status/JobName
# from stdin (piped from aws sagemaker list-training-jobs) and prints with duration and
# estimated on-demand cost columns. Cost requires AWS_REGION env var and instance type
# encoded in the job name (set automatically by sm-submit).

import sys
import os
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_cost import estimate_cost  # noqa: E402  (path manipulation above is intentional)

region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""


def extract_inst_enc(name):
    """Extract encoded instance type from job name, or None if not present.

    Job name format (new): {prefix}{script}-{inst_enc}-{YYYYMMDD}-{HHMMSS}
    Job name format (old): {prefix}{script}-{YYYYMMDD}-{HHMMSS}
    The encoded instance sits between the first hyphen and the date suffix;
    absent for jobs submitted before instance-type encoding was added.
    """
    base = re.sub(r'-\d{8}-\d{6}$', '', name)
    idx = base.find('-')
    if idx == -1:
        return None, False
    middle = base[idx + 1:]
    if '-' not in middle:
        return None, False
    is_spot = middle.endswith('-spot')
    if is_spot:
        middle = middle[:-5]
    return middle, is_spot


print(f"{'Created':<19}  {'Duration':<15}  {'Status':<12}  {'Est Cost':<10}  Job")
print("-" * 100)
for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 4:
        continue
    created, end_time, status, name = parts
    hrs = None
    try:
        c = datetime.fromisoformat(created)
        if end_time != "None":
            e = datetime.fromisoformat(end_time)
            hrs = (e - c).total_seconds() / 3600
            dur = f"{hrs:.2f}h ({hrs * 60:6.1f}m)"
        else:
            dur = "running"
    except Exception:
        dur = "?"

    inst_enc, is_spot = extract_inst_enc(name)
    if hrs is not None:
        cost, _ = estimate_cost(region, inst_enc, hrs)
        if cost is not None:
            cost_str = f"{'<=' if is_spot else ''}${cost:.2f}"
        else:
            cost_str = "unknown"
    elif end_time == "None":
        cost_str = "(running)"
    else:
        cost_str = "unknown"

    print(f"{created[:19]}  {dur:<15}  {status:<12}  {cost_str:<10}  {name}")
