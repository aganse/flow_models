# AWS Batch job submission

This directory contains support files for submitting training jobs via AWS Batch.
The Makefile targets below are available from the repo root directory.

See [`sagemaker-support/README.md`](../sagemaker-support/README.md) for the
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
make run-build                 # trigger a build; pushes to ECR tagged with git hash + latest
```

To push a locally-built image instead:

```bash
make build-gpu                 # build GPU image locally
make push-to-ecr DEVICE=gpu    # push to ECR
```

## Submitting a job

```bash
make run-batchjob              # submit job; returns a JOBID immediately
```

## Monitoring jobs

```bash
make list-jobs                 # show all jobs across all statuses
make list-job-status JOBID=<id>  # check status of a specific job
make cancel-job JOBID=<id>     # cancel a running job
```

## Resource inspection and cleanup

```bash
make list-ecr-repos            # list ECR repos and images
make list-compute-resources    # list compute envs, queues, job defs, EC2 instances
make delete-compute-resources1 # disable and delete job queue + job defs (step 1)
make delete-compute-resources2 # delete compute environment (step 2, after step 1 settles)
```
