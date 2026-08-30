#!/bin/bash
set -e

SCRIPT="train_flowmodels${TRAINING_SCRIPT:-1}.py"
HP_FILE="/opt/ml/input/config/hyperparameters.json"
PARAMS_FILE="/opt/ml/input/data/params/params${TRAINING_SCRIPT:-1}.json"

# For local Docker: if params file is mounted but HP file is absent, create it
# so the script's param-loading logic works the same in all container contexts.
if [ ! -f "$HP_FILE" ] && [ -f "$PARAMS_FILE" ]; then
    mkdir -p "$(dirname "$HP_FILE")"
    BLOB=$(base64 -w0 "$PARAMS_FILE" 2>/dev/null || base64 "$PARAMS_FILE")
    printf '{"params": "%s"}' "$BLOB" > "$HP_FILE"
fi

echo "Running: python3 $SCRIPT"
exec python3 /app/$SCRIPT
