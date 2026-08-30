#!/bin/bash
# Estimate cost of a completed SageMaker training job from instance type and duration.
# Called from sm-status in makefile-sagemaker.mk.
# Usage: sm-cost.sh REGION INSTANCE_TYPE START_TIME END_TIME IS_SPOT
#
# On-demand rates (USD/hr) sourced from: https://aws.amazon.com/sagemaker/pricing/
# Covers US regions only; us-east-1, us-east-2, and us-west-2 share the same price tier.
# us-west-1 (N. California) is typically ~25% higher.
# Update rates here when AWS adjusts pricing (changes infrequently).

REGION="${1:-}"
INSTANCE="${2:-}"
START="${3:-}"
END="${4:-}"
IS_SPOT="${5:-}"

if [ -z "$INSTANCE" ] || [ -z "$START" ]; then
    echo "Estimated cost: unavailable"
    exit 0
fi

if [ -z "$END" ] || [ "$END" = "None" ]; then
    echo "Estimated cost: in progress (job not yet complete)"
    exit 0
fi

# On-demand price lookup: REGION:INSTANCE_TYPE -> USD/hr
case "${REGION}:${INSTANCE}" in
    # g4dn (T4 GPU) -------------------------------------------------------
    us-east-1:ml.g4dn.xlarge|us-east-2:ml.g4dn.xlarge|us-west-2:ml.g4dn.xlarge)       RATE=0.7364 ;;
    us-west-1:ml.g4dn.xlarge)                                                            RATE=0.9206 ;;
    us-east-1:ml.g4dn.2xlarge|us-east-2:ml.g4dn.2xlarge|us-west-2:ml.g4dn.2xlarge)    RATE=1.4726 ;;
    us-west-1:ml.g4dn.2xlarge)                                                           RATE=1.8408 ;;
    us-east-1:ml.g4dn.4xlarge|us-east-2:ml.g4dn.4xlarge|us-west-2:ml.g4dn.4xlarge)    RATE=2.9452 ;;
    us-west-1:ml.g4dn.4xlarge)                                                           RATE=3.6815 ;;
    us-east-1:ml.g4dn.8xlarge|us-east-2:ml.g4dn.8xlarge|us-west-2:ml.g4dn.8xlarge)    RATE=5.8906 ;;
    us-west-1:ml.g4dn.8xlarge)                                                           RATE=7.3633 ;;
    us-east-1:ml.g4dn.12xlarge|us-east-2:ml.g4dn.12xlarge|us-west-2:ml.g4dn.12xlarge) RATE=8.8358 ;;
    us-west-1:ml.g4dn.12xlarge)                                                          RATE=11.0448 ;;
    us-east-1:ml.g4dn.16xlarge|us-east-2:ml.g4dn.16xlarge|us-west-2:ml.g4dn.16xlarge) RATE=11.7813 ;;
    us-west-1:ml.g4dn.16xlarge)                                                          RATE=14.7266 ;;
    # p3 (V100 GPU) -------------------------------------------------------
    us-east-1:ml.p3.2xlarge|us-east-2:ml.p3.2xlarge|us-west-2:ml.p3.2xlarge)          RATE=3.825 ;;
    us-west-1:ml.p3.2xlarge)                                                             RATE=4.7813 ;;
    us-east-1:ml.p3.8xlarge|us-east-2:ml.p3.8xlarge|us-west-2:ml.p3.8xlarge)          RATE=15.301 ;;
    us-west-1:ml.p3.8xlarge)                                                             RATE=19.126 ;;
    us-east-1:ml.p3.16xlarge|us-east-2:ml.p3.16xlarge|us-west-2:ml.p3.16xlarge)       RATE=30.602 ;;
    us-west-1:ml.p3.16xlarge)                                                            RATE=38.253 ;;
    # m5 (CPU) ------------------------------------------------------------
    us-east-1:ml.m5.large|us-east-2:ml.m5.large|us-west-2:ml.m5.large)                RATE=0.1342 ;;
    us-west-1:ml.m5.large)                                                               RATE=0.1678 ;;
    us-east-1:ml.m5.xlarge|us-east-2:ml.m5.xlarge|us-west-2:ml.m5.xlarge)             RATE=0.2683 ;;
    us-west-1:ml.m5.xlarge)                                                              RATE=0.3354 ;;
    us-east-1:ml.m5.2xlarge|us-east-2:ml.m5.2xlarge|us-west-2:ml.m5.2xlarge)          RATE=0.5366 ;;
    us-west-1:ml.m5.2xlarge)                                                             RATE=0.6708 ;;
    us-east-1:ml.m5.4xlarge|us-east-2:ml.m5.4xlarge|us-west-2:ml.m5.4xlarge)          RATE=1.0733 ;;
    us-west-1:ml.m5.4xlarge)                                                             RATE=1.3416 ;;
    *)
        echo "Estimated cost: rate unknown for ${INSTANCE} in ${REGION} — see https://aws.amazon.com/sagemaker/pricing/"
        exit 0
        ;;
esac

python3 - "${START}" "${END}" "${RATE}" "${IS_SPOT}" <<'PYEOF'
import sys
from datetime import datetime

start_str, end_str, rate_str, is_spot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    start = datetime.fromisoformat(start_str)
    end   = datetime.fromisoformat(end_str)
    rate  = float(rate_str)
    hrs   = (end - start).total_seconds() / 3600
    cost  = hrs * rate
    prefix    = "<=" if is_spot == "True" else ""
    spot_note = "  (spot; on-demand rate is upper bound)" if is_spot == "True" else ""
    print(f"Estimated cost: {prefix}${cost:.2f}{spot_note}  ({hrs:.2f} hrs x ${rate:.4f}/hr on-demand)")
except Exception as e:
    print(f"Estimated cost: unavailable ({e})")
PYEOF
