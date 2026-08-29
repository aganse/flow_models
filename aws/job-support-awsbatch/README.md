# AWS Batch job submission

This directory contains support files for submitting training jobs via AWS Batch.
The Makefile targets below are available from the repo root directory.

See [`job-support-sagemaker/README.md`](../job-support-sagemaker/README.md) for the
SageMaker Training Jobs alternative, which is simpler for one-off runs.

For ECR/CodeBuild setup (shared infrastructure), see
[`job-support-common/README.md`](../job-support-common/README.md).

## Required environment variables

Set these in your shell environment (e.g. `~/.zshrc` or `~/.bashrc`):

```bash
export AWS_ACCT_ID=123456789012
export AWS_REGION=us-west-2
export AWSBATCH_SUBNET=$TRAINING_SUBNET     # set from make create-training-subnet (see job-support-common)
export AWSBATCH_SG=$TRAINING_SG            # set from make create-training-sg (see job-support-common)
export MLFLOW_TRACKING_URI=http://10.0.1.50:5000   # MLflow server in your VPC
export IMAGES_PATH=s3://mybucket/afhq               # training data root (train/ and val/ subdirs)
export WEIGHTS_PATH=s3://mybucket/weights           # S3 URI for model weights (optional; only used if save_model_weights=True)
```

`MLFLOW_TRACKING_URI`, `IMAGES_PATH`, and `WEIGHTS_PATH` are injected into the
container at job registration time via `job_definition_template.json`. Re-run
`make register-job-definition` after changing any of these values.

## One-time setup (run once per AWS account)

```bash
# Shared VPC infrastructure (see aws/job-support-common/README.md):
make create-training-subnet VPC_ID=vpc-xxx  # dedicated private training subnet
make create-training-sg VPC_ID=vpc-xxx      # security group for training compute

# Shared ECR/CodeBuild infrastructure (see aws/job-support-common/README.md):
make create-ecr-repo
make create-codebuild-role
make create-codebuild-project

# Batch-specific:
make create-batch-executation-role  # create BatchExecutionRole (required by job definition)
make create-batch-instance-profile  # create IAM instance profile for Batch compute
```

## Occasional setup (run once per compute configuration)

```bash
make create-compute-env        # create Batch compute environment (g4dn.xlarge, scales to 0)
make create-job-queue          # create Batch job queue
make register-job-definition   # register job definition (re-run after image changes)
```

## Building and pushing the Docker image

Image builds are managed via CodeBuild targets in `aws/job-support-common` — see
[`aws/job-support-common/README.md`](../job-support-common/README.md) for full details.

```bash
make build                          # main+gpu -> :main-gpu and :latest
make build BRANCH=myfeature         # myfeature+gpu -> :myfeature-gpu only
make build-status BUILD=<id>        # show status of a submitted build
make build-logs BUILD=<id>          # fetch CloudWatch logs for a build
```

## Submitting a job

```bash
make batch-submit              # submit job; prints JOBID immediately
```

## Monitoring jobs

```bash
make batch-list                   # show all jobs across all statuses
make batch-status JOBID=<id>      # check status of a specific job
make batch-logs JOBID=<id>        # fetch CloudWatch logs for a job
make batch-cancel JOBID=<id>      # cancel a running job
```

The job ID is printed by `make batch-submit` immediately after submission.

## Resource inspection and cleanup

```bash
make list-compute-resources    # list compute envs, queues, job defs, EC2 instances
make delete-compute-resources1 # disable and delete job queue + job defs (step 1)
make delete-compute-resources2 # delete compute environment (step 2, after step 1 settles)
```
