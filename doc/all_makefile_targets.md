# Makefile Targets Reference

All targets are run from the repo root directory.

---

## Root `makefile`

| Target | Description |
|---|---|
| `create-env` | Create a new numbered `.venvN` virtualenv and install `requirements.txt` |
| `install-dev` | Install `requirements-dev.txt` into the active virtualenv (for tests/linting) |
| `unittests` | Run the full test suite via `python -m unittest -v` |
| `lint` | Run `flake8` linting check |
| `local-build [DEVICE=cpu]` | Build Docker image locally; tags as `flow_models:vX.Y.Z-{DEVICE}` |
| `run-local SCRIPT=N [DEVICE=cpu]` | Run training script `N` in local Docker container |

---

## `sagemaker-support/makefile-sagemakersupport.mk`

### One-time / rare
| Target | Description |
|---|---|
| `sm-create-role` | Create `SageMakerExecutionRole` IAM role with S3/ECR/CloudWatch permissions.  Safe to run again if forgot whether already ran it. |
| `sm-list-role` | Check whether `SageMakerExecutionRole` exists and show its ARN and creation date |

### ECR images
| Target | Description |
|---|---|
| `ecr-list-images` | List images in this project's ECR repository with all tags and push dates |
| `ecr-delete-image TAG=<tag>` | Delete an image from ECR by tag |
| `ecr-retag-image FROM=<tag> TO=<tag>` | Rename an ECR image tag (adds new tag, removes old) |

### Build (shared with AWS Batch)
| Target | Description |
|---|---|
| `run-build [BRANCH=main] [DEVICE=gpu]` | Trigger CodeBuild to build and push image to ECR; tags as `{branch}-{device}` (`:latest` updated only for `main-gpu`) |
| `build-status BUILD=<id>` | Show status/phase of a CodeBuild run |
| `build-logs BUILD=<id>` | Fetch CloudWatch logs for a CodeBuild run |

### Training jobs
| Target | Description |
|---|---|
| `sm-run SCRIPT=N [TAG=latest] [SPOT=1] [SM_MAX_WAIT=90000]` | Submit SageMaker Training Job for `train_flowmodelsN.py`; `SPOT=1` enables managed spot training with checkpointing (requires `WEIGHTS_PATH`); `SM_MAX_WAIT` sets total wall-clock deadline in seconds including interruption waits (must be ≥ `SM_MAX_RUNTIME=86400`) |
| `sm-list-jobs` | List recent SageMaker training jobs and their status |
| `sm-status JOB=<name>` | Show status, failure reason, and timing for a specific job |
| `sm-logs JOB=<name>` | Fetch CloudWatch logs for a job |
| `sm-cancel JOB=<name>` | Stop a running job |

---

## `awsbatch-support/makefile-awsbatchsupport.mk`

### One-time / rare
| Target | Description |
|---|---|
| `create-ecr-repo` | Create the ECR repository for Docker images |
| `create-codebuild-role` | Create `CodeBuildServiceRole` IAM role for CodeBuild |
| `create-batch-instance-profile` | Create `BatchInstanceProfile` IAM instance profile for Batch compute nodes |

### Occasional setup
| Target | Description |
|---|---|
| `create-codebuild-project` | Create CodeBuild project linked to this GitHub repo |
| `create-compute-env` | Create Batch managed compute environment (g4dn.xlarge, scales to 0) |
| `create-job-queue` | Create Batch job queue |
| `register-job-definition` | Register Batch job definition (re-run after image or env changes) |

### Build (shared with SageMaker)
| Target | Description |
|---|---|
| `run-build [BRANCH=main] [DEVICE=gpu]` | Trigger CodeBuild to build and push image to ECR; tags as `{branch}-{device}` (`:latest` updated only for `main-gpu`) |
| `build-status BUILD=<id>` | Show status/phase of a CodeBuild run |
| `build-logs BUILD=<id>` | Fetch CloudWatch logs for a CodeBuild run |

### Jobs
| Target | Description |
|---|---|
| `run-batchjob` | Submit a Batch training job; prints job ID immediately |
| `list-jobs` | List all Batch jobs across all statuses |
| `list-job-status JOBID=<id>` | Show detailed status of a specific Batch job |
| `batch-logs JOBID=<id>` | Fetch CloudWatch logs for a Batch job |
| `cancel-job JOBID=<id>` | Cancel a running Batch job |

### Inspection and cleanup
| Target | Description |
|---|---|
| `list-ecr-repos` | List ECR repositories and their images |
| `list-roles` | List policies attached to CodeBuild and Batch IAM roles |
| `list-compute-resources` | List compute envs, job queues, job defs, and running EC2 instances |
| `delete-compute-resources1` | Disable and delete job queue and job definitions (step 1 of 2) |
| `delete-compute-resources2` | Delete compute environment (step 2 of 2, run after step 1 settles) |
| `delete-roles` | Detach policies and delete CodeBuild/Batch IAM roles |
| `push-to-ecr DEVICE=gpu` | Tag and push a locally-built image to ECR (alternative to CodeBuild) |
