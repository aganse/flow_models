# Shared job-support infrastructure

This directory contains support files shared across all cloud training platforms
(SageMaker and AWS Batch). The Makefile targets below are available from the repo
root directory.

- [`job-support-sagemaker/README.md`](../job-support-sagemaker/README.md) — SageMaker Training Jobs (recommended for full single runs)
- [`job-support-awsbatch/README.md`](../job-support-awsbatch/README.md) — AWS Batch (for queued/parallel job sweeps)

## Contents

| File | Purpose |
|---|---|
| `makefile-common.mk` | Shared Makefile targets (included by root `makefile`) |
| `buildspec.yml` | AWS CodeBuild build specification for Docker image builds |
| `codebuild-trust-policy.json` | IAM trust policy for CodeBuildServiceRole |
| `codebuild-cloudwatch-policy.json` | IAM CloudWatch policy for CodeBuildServiceRole |
| `entrypoint.sh` | Docker container entrypoint (shared by SageMaker and local Docker runs) |

## One-time setup

These targets create the shared AWS infrastructure used by both SageMaker and Batch:

```bash
make create-ecr-repo                 # create the ECR repository (very first time only)
make create-codebuild-role           # create CodeBuildServiceRole IAM role
make create-codebuild-project        # create the CodeBuild project
make update-codebuild-project        # update an existing CodeBuild project (e.g. after path changes)
```

```bash
make list-roles                      # list policies on CodeBuild and Batch IAM roles
make delete-roles                    # detach policies and delete CodeBuild/Batch IAM roles
make list-ecr-repos                  # broad view of all ECR repos and images
make push-to-ecr DEVICE=gpu          # push a locally-built image to ECR (alternative to CodeBuild)
```

## VPC infrastructure for training jobs

Both SageMaker and Batch training jobs run in a dedicated private subnet with no
internet access. They reach ECR, S3, and CloudWatch exclusively via VPC endpoints,
keeping all traffic on the AWS backbone. MLflow is reachable at its private VPC IP.

### One-time VPC setup

```bash
make create-training-subnet VPC_ID=vpc-xxx        # create private subnet + route table
                                                   # → export TRAINING_SUBNET=subnet-xxx
make create-training-sg VPC_ID=vpc-xxx            # create compute security group
                                                   # → export TRAINING_SG=sg-xxx
```

After running both, set these in your shell environment:

```bash
export TRAINING_SUBNET=subnet-xxx
export TRAINING_SG=sg-xxx
export SM_SUBNET=$TRAINING_SUBNET      # used by sm-submit
export SM_SG=$TRAINING_SG             # used by sm-submit
export AWSBATCH_SUBNET=$TRAINING_SUBNET  # used by create-compute-env
export AWSBATCH_SG=$TRAINING_SG          # used by create-compute-env
```

Also add an inbound rule on your MLflow EC2's security group: allow TCP port 5000
from `TRAINING_SG` so training instances can reach the MLflow tracking server.

### Per-run VPC endpoints

The three interface endpoints (~$0.01/AZ/hr each) should be created before a
training run and deleted after to avoid unnecessary cost. The S3 gateway endpoint
is free and permanent.

```bash
make create-vpc-endpoints VPC_ID=vpc-xxx  # create before submitting jobs
make sm-submit SCRIPT=N                   # (or batch-submit)
make sm-status JOB=...                    # monitor until Completed/Failed
make delete-vpc-endpoints                 # delete interface endpoints after run
```

## Local Docker builds

Build and run the training image locally without a cloud submission:

```bash
make local-build DEVICE=cpu         # build CPU image locally (also =gpu on GPU-equipped machine)
make local-run SCRIPT=N DEVICE=cpu  # run training script N in the local container
```

`make local-build` uses the local working directory as the build context, so it
picks up any uncommitted `.py` file changes — useful for iterative testing before
pushing to GitHub.

Required environment variable for `local-run`:
- `MLFLOW_TRACKING_URI`: MLflow server address as seen from *inside* the container.
  On macOS use `http://192.168.65.254:5000`; on Linux use `http://172.17.0.1:5000`.

## CodeBuild / ECR image management

Image builds run on AWS CodeBuild, which pulls from GitHub and pushes to ECR.
Both SageMaker and AWS Batch use the same ECR image.

### Building

```bash
make build                           # build main branch, gpu -> :main-gpu and :latest
make build BRANCH=myfeature          # build feature branch -> :myfeature-gpu only
make build BRANCH=myfeature DEVICE=cpu  # -> :myfeature-cpu only
```

`BRANCH` defaults to `main`, `DEVICE` defaults to `gpu`. `:latest` always points
to the most recent `main-gpu` build and is never updated by branch builds.

Code must be pushed to GitHub before running `make build` — CodeBuild pulls
directly from the repo at the specified branch.

```bash
make build-status BUILD=<id>   # show status of a submitted build
make build-logs BUILD=<id>     # fetch CloudWatch logs for a build
```

The build ID is printed by `make build` immediately after submission.

### ECR image inspection

```bash
make list-ecr-images                      # list all images with tags and push dates
make delete-ecr-image TAG=mybranch-gpu    # delete an image by tag
make tag-ecr-image FROM=old TO=new        # tag an image (like docker tag; FROM left intact)
```

## Required environment variables

```bash
export AWS_ACCT_ID=123456789012
export AWS_REGION=us-west-2
```

These are required by `make build`, `make ecr-*`, and all cloud submission targets.
`MLFLOW_TRACKING_URI`, `IMAGES_PATH`, and `WEIGHTS_PATH` are additionally needed
for `make local-run` and the cloud submission targets in the platform-specific makefiles.
