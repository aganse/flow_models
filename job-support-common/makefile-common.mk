# Shared makefile for local Docker builds and cloud image/build infrastructure.
# Included from root Makefile; targets here apply regardless of cloud platform.
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID   - AWS account ID
#   AWS_REGION    - AWS region (e.g. us-west-2)
#   IMAGES_PATH   - training data location (local path or s3:// URI)
#   WEIGHTS_PATH  - model weights location (local path or s3:// URI, optional)
#   MLFLOW_TRACKING_URI - MLflow server URI


# Repo/project settings — adjust to taste:
ECR_REPO=flow_models
CODEBUILD_PROJ=flow_models_build
BRANCH ?= main  # git branch to build from; override with e.g. make build BRANCH=myfeature
DEVICE ?= gpu   # cpu or gpu; override with e.g. make local-build DEVICE=cpu

export ECR_REPO_URI=${AWS_ACCT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}


# ── Local Docker ──────────────────────────────────────────────────────────────

local-build:
	# Build image locally for dev/testing.  Usage: make local-build [DEVICE=cpu]
	docker build -t $(ECR_REPO):$(version)-$(DEVICE) .

local-run:
	# Run a training script locally in Docker.
	# Usage: make local-run SCRIPT=N [DEVICE=cpu]
ifndef SCRIPT
	@echo "Usage: make local-run SCRIPT=1  (or SCRIPT=2, etc.)"
	@exit 1
endif
ifndef MLFLOW_TRACKING_URI
	$(error MLFLOW_TRACKING_URI is not set. You must export it before running make local-run)
endif
ifneq ($(SCRIPT),1)
ifeq ($(IMAGES_PATH),)
	$(error IMAGES_PATH is not set. Required for SCRIPT=$(SCRIPT) runs)
endif
endif
	$(eval DATA_MOUNT := $(shell \
	  if [ -z "$(IMAGES_PATH)" ]; then echo ""; \
	  elif echo "$(IMAGES_PATH)" | grep -q "^s3://"; then echo ""; \
	  else echo "-v $(IMAGES_PATH):$(IMAGES_PATH)"; \
	  fi))
	$(eval AWS_MOUNT := $(shell \
	  [ -d $(HOME)/.aws ] && echo "-v $(HOME)/.aws:/root/.aws" || echo ""))
	docker run --rm -it \
	  -e TRAINING_SCRIPT=$(SCRIPT) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	  -e IMAGES_PATH=$(IMAGES_PATH) \
	  -e WEIGHTS_PATH=$(WEIGHTS_PATH) \
	  -e HOST_USER=$(shell whoami) \
      -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
      -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
      -e AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
	  $(DATA_MOUNT) \
	  $(AWS_MOUNT) \
	  -v $(PWD)/params:/opt/ml/input/data/params \
	  -v /usr/local/mlruns:/usr/local/mlruns \
	  $(ECR_REPO):$(version)-$(DEVICE)


# ── CodeBuild / ECR ───────────────────────────────────────────────────────────

build:
	# Build and push Docker image to ECR via CodeBuild.
	# Usage: make build [BRANCH=main] [DEVICE=gpu]
	# Examples:
	#   make build                          # main+gpu -> :main-gpu and :latest
	#   make build BRANCH=myfeature         # myfeature+gpu -> :myfeature-gpu only
	#   make build BRANCH=myfeature DEVICE=cpu  # -> :myfeature-cpu only
ifndef AWS_ACCT_ID
	$(error AWS_ACCT_ID is not set. Export it before running build)
endif
ifndef AWS_REGION
	$(error AWS_REGION is not set. Export it before running build)
endif
	$(eval BUILD_ID := $(shell aws codebuild start-build --project-name ${CODEBUILD_PROJ} \
	  --source-version $(BRANCH) \
	  --environment-variables-override \
	    name=DEVICE,value=$(DEVICE),type=PLAINTEXT \
	    name=BRANCH_TAG,value=$(BRANCH),type=PLAINTEXT \
	  --query 'build.id' --output text))
	@echo "Submitted build: $(BUILD_ID)"
	@echo "Check status: make build-status BUILD=$(BUILD_ID)"
	@echo "Check logs:   make build-logs BUILD=$(BUILD_ID)"

build-status:
	# Show status of a submitted CodeBuild run.
	# Usage: make build-status BUILD=<id-from-build>
ifndef BUILD
	@echo "Usage: make build-status BUILD=<id-from-build>"
	@echo
	@exit 1
endif
	@aws codebuild batch-get-builds --ids $(BUILD) \
	  --query 'builds[0].{Status:buildStatus,Phase:currentPhase,Start:startTime,End:endTime}' \
	  --output table --no-cli-pager

build-logs:
	# Fetch CloudWatch logs for a CodeBuild run.
	# Usage: make build-logs BUILD=<id-from-build>
ifndef BUILD
	@echo "Usage: make build-logs BUILD=<id-from-build>"
	@echo
	@exit 1
endif
	@GROUP=$$(aws codebuild batch-get-builds --ids $(BUILD) \
	  --query 'builds[0].logs.groupName' --output text 2>/dev/null); \
	STREAM=$$(aws codebuild batch-get-builds --ids $(BUILD) \
	  --query 'builds[0].logs.streamName' --output text 2>/dev/null); \
	if [ -z "$$GROUP" ] || [ "$$GROUP" = "None" ]; then \
	  echo "No logs found yet for build $(BUILD)"; \
	else \
	  aws logs get-log-events \
	    --log-group-name "$$GROUP" \
	    --log-stream-name "$$STREAM" \
	    --query 'events[*].message' --output text --no-cli-pager; \
	fi

ecr-list-images:
	# List images in this project's ECR repository with tags and push dates.
	@aws ecr describe-images --repository-name $(ECR_REPO) --no-cli-pager \
	  --query 'sort_by(imageDetails,&imagePushedAt)[*].{PushedAt:imagePushedAt,Tags:join(`,`,imageTags),SizeBytes:imageSizeInBytes}' \
	  --output table 2>/dev/null || echo "No images found (repo may not exist yet)."

ecr-delete-image:
	# Delete an image from ECR by tag.  Usage: make ecr-delete-image TAG=mybranch-gpu
ifndef TAG
	@echo "Usage: make ecr-delete-image TAG=mybranch-gpu"
	@exit 1
endif
	@aws ecr batch-delete-image --repository-name $(ECR_REPO) \
	  --image-ids imageTag=$(TAG) --no-cli-pager
	@echo "Deleted: $(TAG)"

ecr-tag-image:
	# Tag an ECR image — equivalent to `docker tag FROM TO`.
	# ECR atomically moves TO if it already exists on another image.
	# FROM tag is left intact.  Usage: make ecr-tag-image FROM=src-tag TO=dst-tag
ifndef FROM
	@echo "Usage: make ecr-tag-image FROM=src-tag TO=dst-tag"
	@exit 1
endif
ifndef TO
	@echo "Usage: make ecr-tag-image FROM=src-tag TO=dst-tag"
	@exit 1
endif
	@MANIFEST=$$(aws ecr batch-get-image --repository-name $(ECR_REPO) \
	  --image-ids imageTag=$(FROM) \
	  --query 'images[0].imageManifest' --output text); \
	aws ecr put-image --repository-name $(ECR_REPO) \
	  --image-tag $(TO) --image-manifest "$$MANIFEST" --no-cli-pager > /dev/null
	@echo "Tagged: $(FROM) -> $(TO)  ($(FROM) tag unchanged)"


.PHONY: local-build local-run build build-status build-logs \
	ecr-list-images ecr-delete-image ecr-tag-image
