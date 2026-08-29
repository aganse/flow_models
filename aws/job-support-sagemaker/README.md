# SageMaker Training Jobs submission

This directory contains support files for submitting training jobs via AWS
SageMaker Training Jobs. The Makefile targets below are available from the
repo root directory.

See [`job-support-awsbatch/README.md`](../job-support-awsbatch/README.md) for the
AWS Batch alternative, which is better suited for queued/parallel job sweeps.

## Required environment variables

Set these in your shell environment (e.g. `~/.zshrc` or `~/.bashrc`):

```bash
export AWS_ACCT_ID=123456789012
export AWS_REGION=us-west-2
export SM_SUBNET=subnet-xxxxxxxxxxxxxxxxx
export SM_SG=sg-xxxxxxxxxxxxxxxxx
export IMAGES_PATH=s3://mybucket/afhq       # S3 URI of training data (must be S3 for SageMaker)
export WEIGHTS_PATH=s3://mybucket/weights   # S3 URI for model weights (optional; only used if save_model_weights=True)
export MLFLOW_TRACKING_URI=http://10.0.1.50:5000   # MLflow server in your VPC
```

## One-time setup (run once per AWS account)

```bash
make create-ecr-repo    # create ECR repo for Docker images (shared with Batch)
make sm-create-role     # create SageMaker execution IAM role
```

## Building and pushing the Docker image

Image builds run on AWS CodeBuild (shared with Batch pipeline):

```bash
make build                          # main+gpu -> :main-gpu and :latest
make build BRANCH=myfeature         # myfeature+gpu -> :myfeature-gpu only
make build BRANCH=myfeature DEVICE=cpu  # -> :myfeature-cpu only
```

`BRANCH` defaults to `main`, `DEVICE` defaults to `gpu`. `:latest` always
points to the most recent `main-gpu` build and is never updated by branch
builds. The same Docker image is used for both SageMaker and AWS Batch runs.

```bash
make build-status BUILD=<id>   # show status of a submitted build
make build-logs BUILD=<id>     # fetch CloudWatch logs for a build
```

The build ID is printed by `make build` immediately after submission.

## Configuring hyperparameters

Edit the params file for the script you intend to run before submitting:

```
params/params1.json    # parameters for train_flowmodels1.py
params/params2.json    # parameters for train_flowmodels2.py
```

These files use the same three-group structure as the training scripts
(`run_params`, `training_params`, `model_arch_params`). Environment-specific
values (`images_path`, MLflow URI) are injected separately via environment
variables and should not appear in these files.

Key `training_params` entries relevant to cloud runs:

| Key | Default | Meaning |
|---|---|---|
| `checkpoint_every_n_epochs` | 5 | Save a checkpoint every N epochs; 0 disables |
| `save_model_weights` | false | Upload final weights to `WEIGHTS_PATH` after training |

## Submitting a job

```bash
make sm-submit SCRIPT=1                    # submit train_flowmodels1.py using :latest
make sm-submit SCRIPT=2                    # submit train_flowmodels2.py using :latest
make sm-submit SCRIPT=2 TAG=myfeature-gpu  # use a specific image tag
```

`TAG` defaults to `latest` (= `main-gpu`).

The job runs on `ml.g4dn.xlarge`, reads training data from `IMAGES_PATH` via
FastFile mode (on-demand S3 streaming, no full copy), logs metrics to the
MLflow server at `MLFLOW_TRACKING_URI`, and terminates automatically on
completion. Inside the container `IMAGES_PATH` is set to the FastFile mount
point `/opt/ml/input/data/training`. If `save_model_weights=True` in
`training_params`, the weights file is uploaded directly to `WEIGHTS_PATH`
via boto3.

## Spot instances (optional)

SageMaker managed spot training uses spare EC2 capacity at up to 90% off
on-demand prices. If the instance is interrupted, SageMaker automatically
saves `/opt/ml/checkpoints/` to S3 and restores it when a new instance
becomes available, so training resumes from the last checkpoint rather than
restarting from epoch 0.

```bash
make sm-submit SCRIPT=2 SPOT=1
```

| Variable | Default | Meaning |
|---|---|---|
| `SM_MAX_WAIT` | 90000 s | Total wall-clock deadline including interruption waits (must be ≥ `SM_MAX_RUNTIME`); override with e.g. `make sm-submit SPOT=1 SM_MAX_WAIT=36000` |

Checkpoint frequency is set by `checkpoint_every_n_epochs` in the params file
(default 5). Checkpoints are also written on non-spot runs, providing
epoch-level recovery from crashes.

> **Note:** AWS Batch spot support is configured at the compute environment
> level (`make create-compute-env`) and does not use this checkpoint mechanism.

## Monitoring jobs

```bash
make sm-list                  # list recent jobs and status
make sm-status JOB=flowmodels2-... # describe a specific job
make sm-logs JOB=flowmodels2-...   # fetch CloudWatch logs for a job
make sm-cancel JOB=flowmodels2-... # stop a running job
```

The job name is printed by `make sm-submit` and follows the pattern
`flowmodelsN-YYYYMMDD-HHMMSS`.
