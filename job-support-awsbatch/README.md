# AWS Batch job submission

This directory contains support files for submitting training jobs via AWS Batch.
The Makefile targets below are available from the repo root directory.

See [`job-support-sagemaker/README.md`](../job-support-sagemaker/README.md) for the
SageMaker Training Jobs alternative, which is simpler for one-off runs.

## Required environment variables

Set these in your shell environment (e.g. `~/.zshrc` or `~/.bashrc`):

```bash
export AWS_ACCT_ID=123456789012
export AWS_REGION=us-west-2
export AWSBATCH_SUBNET=subnet-xxxxxxxxxxxxxxxxx
export AWSBATCH_SG=sg-xxxxxxxxxxxxxxxxx
export MLFLOW_TRACKING_URI=http://10.0.1.50:5000   # MLflow server in your VPC
export IMAGES_PATH=s3://mybucket/afhq               # training data root (train/ and val/ subdirs)
export WEIGHTS_PATH=s3://mybucket/weights           # S3 URI for model weights (optional; only used if save_model_weights=True)
```

`MLFLOW_TRACKING_URI`, `IMAGES_PATH`, and `WEIGHTS_PATH` are injected into the
container at job registration time via `job_definition_template.json`. Re-run
`make register-job-definition` after changing any of these values.

## One-time setup (run once per AWS account)

```bash
make create-ecr-repo           # create ECR repo for Docker images
make create-codebuild-role     # create IAM role for CodeBuild image builds
make create-batch-instance-profile  # create IAM instance profile for Batch compute
```

## Occasional setup (run once per compute configuration)

```bash
make create-compute-env        # create Batch compute environment (g4dn.xlarge, scales to 0)
make create-job-queue          # create Batch job queue
make register-job-definition   # register job definition (re-run after image changes)
```

## Building and pushing the Docker image

Image builds run on AWS CodeBuild (faster than local builds, avoids large upload):

```bash
make create-codebuild-project  # one-time: set up CodeBuild project linked to this repo
make build                          # main+gpu -> :main-gpu and :latest
make build BRANCH=myfeature         # myfeature+gpu -> :myfeature-gpu only
make build BRANCH=myfeature DEVICE=cpu  # -> :myfeature-cpu only
```

`BRANCH` defaults to `main`, `DEVICE` defaults to `gpu`. `:latest` always
points to the most recent `main-gpu` build and is never updated by branch builds.

```bash
make build-status BUILD=<id>   # show status of a submitted build
make build-logs BUILD=<id>     # fetch CloudWatch logs for a build
```

The build ID is printed by `make build` immediately after submission.

To push a locally-built image instead:

```bash
make local-build DEVICE=gpu    # build GPU image locally
make push-to-ecr DEVICE=gpu    # push to ECR
```

## Submitting a job

```bash
make batch-submit              # submit job; prints JOBID immediately
```

## Monitoring jobs

```bash
make batch-list                   # show all jobs across all statuses
make batch-status JOBID=<id>  # check status of a specific job
make batch-logs JOBID=<id>       # fetch CloudWatch logs for a job
make batch-cancel JOBID=<id>       # cancel a running job
```

The job ID is printed by `make batch-submit` immediately after submission.

## Resource inspection and cleanup

```bash
make list-ecr-repos            # list ECR repos and images
make list-compute-resources    # list compute envs, queues, job defs, EC2 instances
make delete-compute-resources1 # disable and delete job queue + job defs (step 1)
make delete-compute-resources2 # delete compute environment (step 2, after step 1 settles)
```
