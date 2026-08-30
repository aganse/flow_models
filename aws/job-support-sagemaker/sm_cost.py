#!/usr/bin/env python3
"""
Estimate SageMaker training job cost from instance type and duration.
Importable as a module (estimate_cost, RATES) or callable as a CLI script.

CLI usage (called from sm-status in makefile-sagemaker.mk):
  sm_cost.py REGION INSTANCE_TYPE START_TIME END_TIME IS_SPOT
  e.g.: sm_cost.py us-west-2 ml.g4dn.xlarge 2024-01-01T12:00:00 2024-01-01T14:00:00 False

On-demand rates (USD/hr) from https://aws.amazon.com/sagemaker/pricing/
Covers US regions only. Update rates here when AWS adjusts pricing (changes infrequently).
"""

import sys
import re
from datetime import datetime

# On-demand rates (USD/hr) keyed by "region:encoded_instance".
# Encoded form: strip "ml." prefix, abbreviate size suffix
#   xlarge→xl, 2xlarge→2xl, 4xlarge→4xl, ..., large→l, medium→m, small→s
RATES = {
    # g4dn (T4 GPU)
    "us-east-1:g4dn-xl":    0.7364, "us-east-2:g4dn-xl":    0.7364,
    "us-west-2:g4dn-xl":    0.7364, "us-west-1:g4dn-xl":    0.9206,
    "us-east-1:g4dn-2xl":   1.4726, "us-east-2:g4dn-2xl":   1.4726,
    "us-west-2:g4dn-2xl":   1.4726, "us-west-1:g4dn-2xl":   1.8408,
    "us-east-1:g4dn-4xl":   2.9452, "us-east-2:g4dn-4xl":   2.9452,
    "us-west-2:g4dn-4xl":   2.9452, "us-west-1:g4dn-4xl":   3.6815,
    "us-east-1:g4dn-8xl":   5.8906, "us-east-2:g4dn-8xl":   5.8906,
    "us-west-2:g4dn-8xl":   5.8906, "us-west-1:g4dn-8xl":   7.3633,
    "us-east-1:g4dn-12xl":  8.8358, "us-east-2:g4dn-12xl":  8.8358,
    "us-west-2:g4dn-12xl":  8.8358, "us-west-1:g4dn-12xl":  11.0448,
    "us-east-1:g4dn-16xl": 11.7813, "us-east-2:g4dn-16xl": 11.7813,
    "us-west-2:g4dn-16xl": 11.7813, "us-west-1:g4dn-16xl": 14.7266,
    # p3 (V100 GPU)
    "us-east-1:p3-2xl":   3.825,  "us-east-2:p3-2xl":   3.825,
    "us-west-2:p3-2xl":   3.825,  "us-west-1:p3-2xl":   4.7813,
    "us-east-1:p3-8xl":  15.301,  "us-east-2:p3-8xl":  15.301,
    "us-west-2:p3-8xl":  15.301,  "us-west-1:p3-8xl":  19.126,
    "us-east-1:p3-16xl": 30.602,  "us-east-2:p3-16xl": 30.602,
    "us-west-2:p3-16xl": 30.602,  "us-west-1:p3-16xl": 38.253,
    # m5 (CPU)
    "us-east-1:m5-l":   0.1342, "us-east-2:m5-l":   0.1342,
    "us-west-2:m5-l":   0.1342, "us-west-1:m5-l":   0.1678,
    "us-east-1:m5-xl":  0.2683, "us-east-2:m5-xl":  0.2683,
    "us-west-2:m5-xl":  0.2683, "us-west-1:m5-xl":  0.3354,
    "us-east-1:m5-2xl": 0.5366, "us-east-2:m5-2xl": 0.5366,
    "us-west-2:m5-2xl": 0.5366, "us-west-1:m5-2xl": 0.6708,
    "us-east-1:m5-4xl": 1.0733, "us-east-2:m5-4xl": 1.0733,
    "us-west-2:m5-4xl": 1.0733, "us-west-1:m5-4xl": 1.3416,
    # t3 (CPU, burstable — availability for training jobs varies by region)
    "us-east-1:t3-m":   0.0582, "us-east-2:t3-m":   0.0582,
    "us-west-2:t3-m":   0.0582, "us-west-1:t3-m":   0.0728,
    "us-east-1:t3-l":   0.1163, "us-east-2:t3-l":   0.1163,
    "us-west-2:t3-l":   0.1163, "us-west-1:t3-l":   0.1454,
    "us-east-1:t3-xl":  0.2326, "us-east-2:t3-xl":  0.2326,
    "us-west-2:t3-xl":  0.2326, "us-west-1:t3-xl":  0.2908,
    "us-east-1:t3-2xl": 0.4653, "us-east-2:t3-2xl": 0.4653,
    "us-west-2:t3-2xl": 0.4653, "us-west-1:t3-2xl": 0.5816,
}

_SIZE_MAP = {
    "xlarge": "xl", "2xlarge": "2xl", "4xlarge": "4xl",
    "8xlarge": "8xl", "12xlarge": "12xl", "16xlarge": "16xl",
    "large": "l", "medium": "m", "small": "s",
}


def encode_instance(full_inst):
    """Encode ml.g4dn.xlarge → g4dn-xl for RATES lookup."""
    inst = re.sub(r'^ml\.', '', full_inst)
    dot = inst.find('.')
    if dot == -1:
        return inst
    family, size = inst[:dot], inst[dot + 1:]
    return f"{family}-{_SIZE_MAP.get(size, size)}"


def estimate_cost(region, inst_enc, hrs):
    """Return (cost, rate) or (None, None) if region/instance not in table."""
    rate = RATES.get(f"{region}:{inst_enc}")
    if rate is None:
        return None, None
    return hrs * rate, rate


if __name__ == "__main__":
    region   = sys.argv[1] if len(sys.argv) > 1 else ""
    instance = sys.argv[2] if len(sys.argv) > 2 else ""
    start    = sys.argv[3] if len(sys.argv) > 3 else ""
    end      = sys.argv[4] if len(sys.argv) > 4 else ""
    is_spot  = sys.argv[5] if len(sys.argv) > 5 else ""

    if not instance or not start:
        print("Estimated cost: unavailable")
        sys.exit(0)

    if not end or end == "None":
        print("Estimated cost: in progress (job not yet complete)")
        sys.exit(0)

    inst_enc = encode_instance(instance)
    cost, rate = estimate_cost(region, inst_enc, 0)  # rate probe
    if rate is None:
        print(f"Estimated cost: rate unknown for {instance} in {region}"
              f" — see https://aws.amazon.com/sagemaker/pricing/")
        sys.exit(0)

    try:
        hrs = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600
        cost = hrs * rate
        prefix    = "<=" if is_spot == "True" else ""
        spot_note = "  (spot; on-demand rate is upper bound)" if is_spot == "True" else ""
        print(f"Estimated cost: {prefix}${cost:.2f}{spot_note}"
              f"  ({hrs:.2f} hrs x ${rate:.4f}/hr on-demand)")
    except Exception as ex:
        print(f"Estimated cost: unavailable ({ex})")
