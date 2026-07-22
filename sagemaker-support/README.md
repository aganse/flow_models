# SageMaker Training Jobs submission

This directory contains support files for submitting training jobs via AWS
SageMaker Training Jobs. The Makefile targets below are available from the
repo root directory.

See [`awsbatch-support/README.md`](../awsbatch-support/README.md) for the
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
make run-build          # trigger CodeBuild; pushes to ECR tagged with git hash + latest
```

The same Docker image is used for both SageMaker and AWS Batch runs.

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

## Submitting a job

```bash
make sm-run SCRIPT=1   # submit train_flowmodels1.py
make sm-run SCRIPT=2   # submit train_flowmodels2.py
```

The job runs on `ml.g4dn.xlarge`, reads training data from `IMAGES_PATH` via
FastFile mode (on-demand S3 streaming, no full copy), logs metrics to the
MLflow server at `MLFLOW_TRACKING_URI`, and terminates automatically on
completion. Inside the container `IMAGES_PATH` is set to the FastFile mount
point `/opt/ml/input/data/training`. If `save_model_weights=True` in
`training_params`, the weights file is uploaded directly to `WEIGHTS_PATH`
via boto3.

## Monitoring jobs

```bash
make sm-list-jobs                  # list recent jobs and status
make sm-status JOB=flowmodels2-... # describe a specific job
make sm-logs JOB=flowmodels2-...   # fetch CloudWatch logs for a job
make sm-cancel JOB=flowmodels2-... # stop a running job
```

The job name is printed by `make sm-run` and follows the pattern
`flowmodelsN-YYYYMMDD-HHMMSS`.
