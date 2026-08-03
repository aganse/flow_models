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

---

## `job-support-common/makefile-common.mk`

### Local Docker
| Target | Description |
|---|---|
| `local-build [DEVICE=cpu]` | Build Docker image locally; tags as `flow_models:vX.Y.Z-{DEVICE}` |
| `local-run SCRIPT=N [DEVICE=cpu]` | Run training script `N` in local Docker container |

### CodeBuild / ECR (shared by SageMaker and AWS Batch)
| Target | Description |
|---|---|
| `build [BRANCH=main] [DEVICE=gpu]` | Trigger CodeBuild to build and push image to ECR; tags as `{branch}-{device}` (`:latest` updated only for `main-gpu`) |
| `build-status BUILD=<id>` | Show status/phase of a CodeBuild run |
| `build-logs BUILD=<id>` | Fetch CloudWatch logs for a CodeBuild run |
| `ecr-list-images` | List images in this project's ECR repository with all tags and push dates |
| `ecr-delete-image TAG=<tag>` | Delete an image from ECR by tag |
| `ecr-tag-image FROM=<tag> TO=<tag>` | Tag an ECR image (equivalent to `docker tag`; `FROM` tag is left intact) |

---

## `job-support-sagemaker/makefile-sagemaker.mk`

### One-time / rare
| Target | Description |
|---|---|
| `sm-create-sg VPC_ID=<vpc-id>` | Create a SageMaker security group (allow-all outbound, no inbound); name encodes creation date as `sagemaker-to-mlflow-YYYY-MM-DD`; prints the SG ID to export as `SM_SG` |
| `sm-create-role` | Create `SageMakerExecutionRole` IAM role with S3/ECR/CloudWatch permissions.  Safe to re-run if unsure whether already done. |
| `sm-list-role` | Check whether `SageMakerExecutionRole` exists and show its ARN and creation date |

### Training jobs
| Target | Description |
|---|---|
| `sm-submit SCRIPT=N [TAG=latest] [SPOT=1] [SM_MAX_WAIT=90000]` | Submit SageMaker Training Job for `train_flowmodelsN.py`; `SPOT=1` enables managed spot training with checkpointing (requires `WEIGHTS_PATH`); `SM_MAX_WAIT` sets total wall-clock deadline in seconds including interruption waits (must be ≥ `SM_MAX_RUNTIME=86400`) |
| `sm-list` | List recent SageMaker training jobs and their status |
| `sm-status JOB=<name>` | Show status, failure reason, and timing for a specific job |
| `sm-logs JOB=<name>` | Fetch CloudWatch logs for a job |
| `sm-cancel JOB=<name>` | Stop a running job |

---

## `job-support-awsbatch/makefile-awsbatch.mk`

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

### Jobs
| Target | Description |
|---|---|
| `batch-submit` | Submit a Batch training job; prints job ID immediately |
| `batch-list` | List all Batch jobs across all statuses |
| `batch-status JOBID=<id>` | Show detailed status of a specific Batch job |
| `batch-logs JOBID=<id>` | Fetch CloudWatch logs for a Batch job |
| `batch-cancel JOBID=<id>` | Cancel a running Batch job |

### Inspection and cleanup
| Target | Description |
|---|---|
| `list-ecr-repos` | List all ECR repositories and their images (broad view across all repos) |
| `list-roles` | List policies attached to CodeBuild and Batch IAM roles |
| `list-compute-resources` | List compute envs, job queues, job defs, and running EC2 instances |
| `delete-compute-resources1` | Disable and delete job queue and job definitions (step 1 of 2) |
| `delete-compute-resources2` | Delete compute environment (step 2 of 2, run after step 1 settles) |
| `delete-roles` | Detach policies and delete CodeBuild/Batch IAM roles |
| `push-to-ecr DEVICE=gpu` | Tag and push a locally-built image to ECR (alternative to CodeBuild) |
