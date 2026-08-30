# Shared makefile for local Docker builds and cloud image/build infrastructure.
# Included from root Makefile; targets here apply regardless of cloud platform.
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID   - AWS account ID
#   AWS_REGION    - AWS region (e.g. us-west-2)
#   IMAGES_PATH   - training data location (local path or s3:// URI)
#   WEIGHTS_PATH  - model weights location (local path or s3:// URI, optional)
#   MLFLOW_TRACKING_URI - MLflow server URI
#   TRAINING_SUBNET - private subnet ID for training compute (create-vpc-endpoints)
#   TRAINING_SG   - security group ID for training compute (create-vpc-endpoints)


# Default CIDR for the private training subnet; override with CIDR=x.x.x.x/xx on command line.
# Verify this does not overlap with existing subnets in your VPC before running create-training-subnet.
CIDR ?= 10.0.10.0/24

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


# ── One-time / ECR and CodeBuild setup ───────────────────────────────────────

create-ecr-repo:
	# Create the ECR repository for this project's Docker images. (one-time/rare)
	@aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

check-codebuild-role-exists:
	# Internal check: exits 1 (skip) if CodeBuildServiceRole already exists.
	@aws iam get-role --role-name CodeBuildServiceRole > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Role CodeBuildServiceRole already exists. Skipping creation."; \
		exit 1; \
	fi

create-codebuild-role: check-codebuild-role-exists
	# Create CodeBuildServiceRole needed for create-codebuild-project. (one-time/rare)
	@echo "Creating role CodeBuildServiceRole..."
	@aws iam create-role \
		--role-name CodeBuildServiceRole --no-cli-pager \
		--assume-role-policy-document file://aws/job-support-common/codebuild-trust-policy.json
	@echo "Role CodeBuildServiceRole created successfully."
	@aws iam create-policy --policy-name CodeBuildCloudWatchPolicy \
		--policy-document file://aws/job-support-common/codebuild-cloudwatch-policy.json
	@aws iam attach-role-policy --role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::${AWS_ACCT_ID}:policy/CodeBuildCloudWatchPolicy
	@echo "CodeBuildCloudWatchPolicy successfully attached to role CodeBuildServiceRole."
	@aws iam attach-role-policy \
		--role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
	@echo "AmazonEC2ContainerRegistryPowerUser successfully attached to role CodeBuildServiceRole."
	@aws iam attach-role-policy \
		--role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess
	@echo "AWSCodeBuildAdminAccess successfully attached to role CodeBuildServiceRole."

create-codebuild-project:
	# Create the CodeBuild project (one-time/rare). Use update-codebuild-project if it already exists.
	@aws codebuild create-project \
	    --name ${CODEBUILD_PROJ} \
		--source "type=GITHUB,location=https://github.com/aganse/flow_models.git,buildspec=aws/job-support-common/buildspec.yml" \
	    --artifacts "type=NO_ARTIFACTS" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL,environmentVariables=[{name='ECR_REPO_URI', value='${ECR_REPO_URI}'},{name='DEVICE', value='${DEVICE}'}]" \
		--service-role arn:aws:iam::${AWS_ACCT_ID}:role/CodeBuildServiceRole

update-codebuild-project:
	# Update the existing CodeBuild project (e.g. after buildspec path changes).
	@aws codebuild update-project \
	    --name ${CODEBUILD_PROJ} \
		--source "type=GITHUB,location=https://github.com/aganse/flow_models.git,buildspec=aws/job-support-common/buildspec.yml" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL,environmentVariables=[{name='ECR_REPO_URI', value='${ECR_REPO_URI}'},{name='DEVICE', value='${DEVICE}'}]" \
		--service-role arn:aws:iam::${AWS_ACCT_ID}:role/CodeBuildServiceRole \
		--no-cli-pager > /dev/null
	@echo "CodeBuild project updated."

list-ecr-repos:
	# List all ECR repositories and their images (broad view across all repos).
	@aws ecr describe-repositories --query 'repositories[*].repositoryName' --output text | \
	while read repo; do echo "ECR Repository $${repo}:"; aws ecr list-images --repository-name "$${repo}" --query 'imageIds[*]' --output text --no-cli-pager; done

push-to-ecr:
	# Tag and push a locally-built image to ECR (alternative to CodeBuild).
	# Usage: make push-to-ecr DEVICE=gpu
ifndef DEVICE
	@echo "Usage: make push-to-ecr DEVICE=gpu  # or DEVICE=cpu"
	@echo
endif
	@docker tag ${ECR_REPO}:${version}-$${DEVICE} ${ECR_REPO_URI}:latest
	@aws ecr get-login-password --region $${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO_URI}
	@docker push ${ECR_REPO_URI}:latest

list-roles:
	# List policies attached to CodeBuildServiceRole and AWSBatchServiceRole.
	@-aws iam list-attached-role-policies --role-name CodeBuildServiceRole --no-cli-pager
	@-aws iam list-attached-role-policies --role-name AWSBatchServiceRole --no-cli-pager

delete-roles:
	# Detach policies and delete CodeBuild/Batch IAM roles.
	@-aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
	@-aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess"
	@-aws iam detach-role-policy --role-name AWSBatchServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSBatchServiceRolePolicy"
	@-aws iam delete-role --role-name CodeBuildServiceRole
	@-aws iam delete-role --role-name AWSBatchServiceRole


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

list-ecr-images:
	# List images in this project's ECR repository with tags and push dates.
	@aws ecr describe-images --repository-name $(ECR_REPO) --no-cli-pager \
	  --query 'reverse(sort_by(imageDetails,&imagePushedAt))[*].{PushedAt:imagePushedAt,Tags:join(`,`,imageTags),SizeBytes:imageSizeInBytes}' \
	  --output table 2>/dev/null || echo "No images found (repo may not exist yet)."

delete-ecr-image:
	# Delete an image from ECR by tag.  Usage: make delete-ecr-image TAG=mybranch-gpu
ifndef TAG
	@echo "Usage: make delete-ecr-image TAG=mybranch-gpu"
	@exit 1
endif
	@aws ecr batch-delete-image --repository-name $(ECR_REPO) \
	  --image-ids imageTag=$(TAG) --no-cli-pager
	@echo "Deleted: $(TAG)"

tag-ecr-image:
	# Tag an ECR image — equivalent to `docker tag FROM TO`.
	# ECR atomically moves TO if it already exists on another image.
	# FROM tag is left intact.  Usage: make tag-ecr-image FROM=src-tag TO=dst-tag
ifndef FROM
	@echo "Usage: make tag-ecr-image FROM=src-tag TO=dst-tag"
	@exit 1
endif
ifndef TO
	@echo "Usage: make tag-ecr-image FROM=src-tag TO=dst-tag"
	@exit 1
endif
	@MANIFEST=$$(aws ecr batch-get-image --repository-name $(ECR_REPO) \
	  --image-ids imageTag=$(FROM) \
	  --query 'images[0].imageManifest' --output text); \
	aws ecr put-image --repository-name $(ECR_REPO) \
	  --image-tag $(TO) --image-manifest "$$MANIFEST" --no-cli-pager > /dev/null
	@echo "Tagged: $(FROM) -> $(TO)  ($(FROM) tag unchanged)"

# ── VPC infrastructure (shared by SageMaker and Batch) ───────────────────────

create-training-subnet:
	# Create a dedicated private subnet for training job compute. (one-time/rare)
	# Creates subnet + its own route table with no internet route (private by design).
	# Usage: make create-training-subnet VPC_ID=vpc-xxxxxxxx [CIDR=10.0.10.0/24]
	# Verify CIDR does not overlap with existing subnets in your VPC before running.
ifndef VPC_ID
	@echo "Usage: make create-training-subnet VPC_ID=vpc-xxxxxxxx [CIDR=10.0.10.0/24]"
	@exit 1
endif
	$(eval SUBNET_ID := $(shell aws ec2 create-subnet \
	  --vpc-id $(VPC_ID) --cidr-block $(CIDR) \
	  --query 'Subnet.SubnetId' --output text --no-cli-pager))
	@aws ec2 create-tags --resources $(SUBNET_ID) \
	  --tags Key=Name,Value=compute-job-submissions --no-cli-pager
	$(eval RT_ID := $(shell aws ec2 create-route-table \
	  --vpc-id $(VPC_ID) \
	  --query 'RouteTable.RouteTableId' --output text --no-cli-pager))
	@aws ec2 create-tags --resources $(RT_ID) \
	  --tags Key=Name,Value=compute-job-submissions --no-cli-pager
	@aws ec2 associate-route-table --subnet-id $(SUBNET_ID) \
	  --route-table-id $(RT_ID) --no-cli-pager > /dev/null
	@echo "Created subnet:      $(SUBNET_ID)  CIDR: $(CIDR)"
	@echo "Created route table: $(RT_ID)  (no internet route — private by design)"
	@echo "Next steps:"
	@echo "  export TRAINING_SUBNET=$(SUBNET_ID)"
	@echo "  export SM_SUBNET=\$$TRAINING_SUBNET  AWSBATCH_SUBNET=\$$TRAINING_SUBNET"

create-training-sg:
	# Create security group for training job compute instances. (one-time/rare)
	# Allow-all outbound (for VPC endpoint access), no inbound. Name encodes creation date.
	# Usage: make create-training-sg VPC_ID=vpc-xxxxxxxx
ifndef VPC_ID
	@echo "Usage: make create-training-sg VPC_ID=vpc-xxxxxxxx"
	@exit 1
endif
	$(eval SG_NAME := compute-job-submissions-$(shell date +%Y-%m-%d))
	$(eval SG_ID := $(shell aws ec2 create-security-group \
	  --group-name $(SG_NAME) \
	  --description "Training job compute instances (SageMaker and Batch)" \
	  --vpc-id $(VPC_ID) \
	  --query 'GroupId' --output text --no-cli-pager))
	@aws ec2 create-tags --resources $(SG_ID) \
	  --tags Key=Name,Value=$(SG_NAME) --no-cli-pager
	@echo "Created: $(SG_ID)  ($(SG_NAME))"
	@echo "Outbound: allow-all (AWS default). Inbound: none."
	@echo "Next steps:"
	@echo "  export TRAINING_SG=$(SG_ID)"
	@echo "  export SM_SG=\$$TRAINING_SG  AWSBATCH_SG=\$$TRAINING_SG"
	@echo "  Add inbound rule on your MLflow EC2's SG: port 5000 from $(SG_ID)"

create-vpc-endpoints:
	# Create VPC endpoints for ECR, S3, and CloudWatch Logs. (run before training jobs)
	# S3 gateway endpoint is free and permanent. Interface endpoints (~$$0.01/AZ/hr each)
	# should be deleted after runs via make delete-vpc-endpoints.
	# Usage: make create-vpc-endpoints VPC_ID=vpc-xxxxxxxx
ifndef VPC_ID
	@echo "Usage: make create-vpc-endpoints VPC_ID=vpc-xxxxxxxx"
	@exit 1
endif
ifndef TRAINING_SUBNET
	$(error TRAINING_SUBNET is not set. Export it before running create-vpc-endpoints)
endif
ifndef TRAINING_SG
	$(error TRAINING_SG is not set. Export it before running create-vpc-endpoints)
endif
	# Ensure VPC has DNS support/hostnames enabled (required for interface endpoints)
	@aws ec2 modify-vpc-attribute --vpc-id $(VPC_ID) \
	  --enable-dns-support '{"Value":true}' 2>/dev/null; true
	@aws ec2 modify-vpc-attribute --vpc-id $(VPC_ID) \
	  --enable-dns-hostnames '{"Value":true}' 2>/dev/null; true
	# Allow HTTPS inbound on training SG so instances can reach endpoint ENIs
	@-aws ec2 authorize-security-group-ingress --group-id ${TRAINING_SG} \
	  --protocol tcp --port 443 --source-group ${TRAINING_SG} \
	  --no-cli-pager > /dev/null 2>&1
	# S3 gateway endpoint (free; routes S3 traffic via AWS backbone, not internet)
	$(eval ROUTE_TABLE_ID := $(shell aws ec2 describe-route-tables \
	  --filters "Name=association.subnet-id,Values=${TRAINING_SUBNET}" \
	  --query 'RouteTables[0].RouteTableId' --output text))
	@aws ec2 create-vpc-endpoint --vpc-endpoint-type Gateway \
	  --service-name com.amazonaws.${AWS_REGION}.s3 \
	  --vpc-id $(VPC_ID) --route-table-ids $(ROUTE_TABLE_ID) \
	  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=compute-job-submissions}]' \
	  --no-cli-pager > /dev/null
	@echo "Created S3 gateway endpoint (permanent, free)"
	# ECR API interface endpoint
	@aws ec2 create-vpc-endpoint --vpc-endpoint-type Interface \
	  --service-name com.amazonaws.${AWS_REGION}.ecr.api \
	  --vpc-id $(VPC_ID) --subnet-ids ${TRAINING_SUBNET} \
	  --security-group-ids ${TRAINING_SG} \
	  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=compute-job-submissions}]' \
	  --no-cli-pager > /dev/null
	@echo "Created ecr.api interface endpoint"
	# ECR DKR interface endpoint
	@aws ec2 create-vpc-endpoint --vpc-endpoint-type Interface \
	  --service-name com.amazonaws.${AWS_REGION}.ecr.dkr \
	  --vpc-id $(VPC_ID) --subnet-ids ${TRAINING_SUBNET} \
	  --security-group-ids ${TRAINING_SG} \
	  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=compute-job-submissions}]' \
	  --no-cli-pager > /dev/null
	@echo "Created ecr.dkr interface endpoint"
	# CloudWatch Logs interface endpoint
	@aws ec2 create-vpc-endpoint --vpc-endpoint-type Interface \
	  --service-name com.amazonaws.${AWS_REGION}.logs \
	  --vpc-id $(VPC_ID) --subnet-ids ${TRAINING_SUBNET} \
	  --security-group-ids ${TRAINING_SG} \
	  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=compute-job-submissions}]' \
	  --no-cli-pager > /dev/null
	@echo "Created logs interface endpoint"
	@echo "Done. Delete interface endpoints after training runs: make delete-vpc-endpoints"

delete-vpc-endpoints:
	# Delete the 3 interface endpoints (ecr.api, ecr.dkr, logs) tagged Name=compute-job-submissions.
	# The S3 gateway endpoint is free and left in place.
	$(eval ENDPOINT_IDS := $(shell aws ec2 describe-vpc-endpoints \
	  --filters "Name=tag:Name,Values=compute-job-submissions" \
	            "Name=vpc-endpoint-type,Values=Interface" \
	            "Name=vpc-endpoint-state,Values=available,pending" \
	  --query 'VpcEndpoints[*].VpcEndpointId' --output text))
	@if [ -z "$(ENDPOINT_IDS)" ]; then \
	  echo "No active interface endpoints tagged Name=compute-job-submissions found."; \
	else \
	  aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $(ENDPOINT_IDS) \
	    --no-cli-pager > /dev/null && \
	  echo "Deleted: $(ENDPOINT_IDS)"; \
	fi


common-help:
	@echo "one-time setup: create-ecr-repo  create-codebuild-role  create-codebuild-project"
	@echo "                create-training-subnet VPC_ID=vpc-xxx  create-training-sg VPC_ID=vpc-xxx"
	@echo "update:         update-codebuild-project"
	@echo "local docker:   local-build [DEVICE=cpu]  local-run SCRIPT=N [DEVICE=cpu]"
	@echo "build:          build [BRANCH=main] [DEVICE=gpu]  build-status BUILD=id  build-logs BUILD=id"
	@echo "ecr images:     list-ecr-images  delete-ecr-image TAG=t  tag-ecr-image FROM=t TO=t"
	@echo "ecr repos:      list-ecr-repos  push-to-ecr DEVICE=gpu"
	@echo "roles:          list-roles  delete-roles"
	@echo "vpc endpoints:  create-vpc-endpoints VPC_ID=vpc-xxx  delete-vpc-endpoints"


.PHONY: create-ecr-repo check-codebuild-role-exists create-codebuild-role \
	create-codebuild-project update-codebuild-project \
	list-ecr-repos push-to-ecr list-roles delete-roles \
	local-build local-run build build-status build-logs \
	list-ecr-images delete-ecr-image tag-ecr-image \
	create-training-subnet create-training-sg \
	create-vpc-endpoints delete-vpc-endpoints common-help
