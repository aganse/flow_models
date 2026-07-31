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
make ecr-list-images                      # list all images with tags and push dates
make ecr-delete-image TAG=mybranch-gpu    # delete an image by tag
make ecr-retag-image FROM=old TO=new      # rename a tag (adds new, removes old)
```

## Required environment variables

```bash
export AWS_ACCT_ID=123456789012
export AWS_REGION=us-west-2
```

These are required by `make build`, `make ecr-*`, and all cloud submission targets.
`MLFLOW_TRACKING_URI`, `IMAGES_PATH`, and `WEIGHTS_PATH` are additionally needed
for `make local-run` and the cloud submission targets in the platform-specific makefiles.
