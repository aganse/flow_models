# SageMaker Training Jobs support makefile
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID   - AWS account ID
#   AWS_REGION    - AWS region (e.g. us-west-2)
#   SM_SUBNET     - VPC subnet ID for SageMaker compute
#   SM_SG         - Security group ID for SageMaker compute
#   IMAGES_PATH   - S3 URI of training data (e.g. s3://mybucket/afhq)
#   WEIGHTS_PATH  - S3 URI for model weights persistence (e.g. s3://mybucket/weights, optional)
#   MLFLOW_TRACKING_URI - MLflow server URI (e.g. http://10.0.1.50:5000)
#
# ECR_REPO_URI and AWS_ACCT_ID/AWS_REGION are shared with
# awsbatch-support/makefile-awsbatchsupport.mk (included from root Makefile).


# Repo-specific, non-sensitive settings — adjust to taste:
SM_INSTANCE_TYPE=ml.g4dn.xlarge
SM_VOLUME_SIZE=50
SM_MAX_RUNTIME=86400
SM_MAX_WAIT ?= 90000
SM_JOB_PREFIX=flowmodels
TAG ?= latest  # ECR image tag to use; override with e.g. make sm-run SCRIPT=2 TAG=cleanup-gpu

# Spot instance flags: set when SPOT=1 is passed on the command line.
# SM_MAX_WAIT is the total wall-clock deadline including interruption waits
# (must be >= SM_MAX_RUNTIME); override with e.g. make sm-run SPOT=1 SM_MAX_WAIT=36000
ifdef SPOT
SM_SPOT_FLAGS := --enable-managed-spot-training \
  --checkpoint-config S3Uri=$(WEIGHTS_PATH)/checkpoints/
SM_STOPPING := --stopping-condition \
  MaxRuntimeInSeconds=$(SM_MAX_RUNTIME),MaxWaitTimeInSeconds=$(SM_MAX_WAIT)
else
SM_SPOT_FLAGS :=
SM_STOPPING := --stopping-condition MaxRuntimeInSeconds=$(SM_MAX_RUNTIME)
endif


# Assembled from env vars (sensitive seeds come from environment):
SM_ROLE_ARN=arn:aws:iam::${AWS_ACCT_ID}:role/SageMakerExecutionRole


sm-what-to-do:
	@echo "once/rarely:   sm-create-role"
	@echo "sometimes:     run-build  (shared CodeBuild pipeline, pushes to ECR)"
	@echo "more often:    sm-run SCRIPT=1 [TAG=main-gpu]  (TAG defaults to latest)"
	@echo "job checks:    sm-list-jobs  sm-status JOB=name  sm-logs JOB=name  sm-cancel JOB=name"

sm-create-role:
	# Create the SageMaker execution role (one-time/rare run).
	# Grants SageMaker permission to pull from ECR, read/write S3, write CloudWatch logs,
	# and access VPC resources.
	@aws iam create-role --role-name SageMakerExecutionRole --no-cli-pager \
		--assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
	@aws iam attach-role-policy --role-name SageMakerExecutionRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
	@aws iam attach-role-policy --role-name SageMakerExecutionRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
	@echo "SageMakerExecutionRole created."

sm-run:
	# Submit a SageMaker Training Job for train_flowmodelsN.py.
	# Usage: make sm-run SCRIPT=1  (or SCRIPT=2, etc.)
ifndef SCRIPT
	@echo "This makefile macro must be called as:"
	@echo "  make sm-run SCRIPT=1   # or SCRIPT=2, etc."
	@echo
	@exit 1
endif
ifndef MLFLOW_TRACKING_URI
	$(error MLFLOW_TRACKING_URI is not set. You must export it before running make run-local)
endif
ifneq ($(SCRIPT),1)
  ifeq ($(filter s3://%,$(IMAGES_PATH)),)
        $(error IMAGES_PATH must be an s3:// URI for SCRIPT=$(SCRIPT) SageMaker runs with images)
  endif
endif
ifneq ($(WEIGHTS_PATH),)
	ifeq ($(filter s3://%,$(WEIGHTS_PATH)),)
	      $(error WEIGHTS_PATH must be an s3:// URI for SageMaker runs if set, got: '$(WEIGHTS_PATH)')
	endif
endif
ifndef AWS_ACCT_ID
	$(error AWS_ACCT_ID is not set. Export it before running sm-run)
endif
ifndef AWS_REGION
	$(error AWS_REGION is not set. Export it before running sm-run)
endif
ifndef SM_SUBNET
	$(error SM_SUBNET is not set. Export it before running sm-run)
endif
ifndef SM_SG
	$(error SM_SG is not set. Export it before running sm-run)
endif
	$(eval JOB_NAME := $(SM_JOB_PREFIX)$(SCRIPT)-$(shell date +%Y%m%d-%H%M%S))
	$(eval PARAMS_BLOB := $(shell base64 -i params/params$(SCRIPT).json 2>/dev/null || base64 params/params$(SCRIPT).json | tr -d '\n'))
	@aws sagemaker create-training-job \
	  --training-job-name $(JOB_NAME) \
	  --algorithm-specification \
	    TrainingImage=$(ECR_REPO_URI):$(TAG),TrainingInputMode=FastFile \
	  --role-arn $(SM_ROLE_ARN) \
	  --input-data-config \
	    '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3Uri":"${IMAGES_PATH}","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
	  --output-data-config S3OutputPath=${WEIGHTS_PATH} \
	  --resource-config \
	    InstanceType=$(SM_INSTANCE_TYPE),InstanceCount=1,VolumeSizeInGB=$(SM_VOLUME_SIZE) \
	  $(SM_SPOT_FLAGS) \
	  $(SM_STOPPING) \
	  --hyper-parameters "{\"params\":\"$(PARAMS_BLOB)\"}" \
	  --environment '{"TRAINING_SCRIPT":"$(SCRIPT)","IMAGE_TAG":"$(TAG)","MLFLOW_TRACKING_URI":"${MLFLOW_TRACKING_URI}","IMAGES_PATH":"/opt/ml/input/data/training","WEIGHTS_PATH":"${WEIGHTS_PATH}"}' \
	  --vpc-config Subnets=${SM_SUBNET},SecurityGroupIds=${SM_SG} \
	  --no-cli-pager
	@echo "Submitted: $(JOB_NAME)"
	@echo "Check status: make sm-status JOB=$(JOB_NAME)"

sm-list-jobs:
	# List recent SageMaker training jobs and their status.
	@aws sagemaker list-training-jobs \
	  --sort-by CreationTime --sort-order Descending \
	  --max-results 20 \
	  --query 'TrainingJobSummaries[*].[CreationTime,TrainingJobStatus,TrainingJobName]' \
	  --output text --no-cli-pager

sm-status:
	# Show status of a submitted job.  Usage: make sm-status JOB=flowmodels2-20240101-120000
ifndef JOB
	@echo "This makefile macro must be called as:"
	@echo "  make sm-status JOB=flowmodels2-20240101-120000  # from output of make sm-run"
	@echo
	@exit 1
endif
	@aws sagemaker describe-training-job --training-job-name ${JOB} \
	  --query '{Status:TrainingJobStatus,Reason:FailureReason,Start:TrainingStartTime,End:TrainingEndTime}' \
	  --output table --no-cli-pager

sm-logs:
	# Tail CloudWatch logs for a running or completed job.
	# Usage: make sm-logs JOB=flowmodels2-20240101-120000
ifndef JOB
	@echo "This makefile macro must be called as:"
	@echo "  make sm-logs JOB=flowmodels2-20240101-120000  # from output of make sm-run"
	@echo
	@exit 1
endif
	@LOG_STREAM=$$(aws logs describe-log-streams \
	  --log-group-name /aws/sagemaker/TrainingJobs \
	  --log-stream-name-prefix ${JOB} \
	  --query 'logStreams[0].logStreamName' --output text 2>/dev/null); \
	if [ "$$LOG_STREAM" = "None" ] || [ -z "$$LOG_STREAM" ]; then \
	  echo "No log stream found yet for job ${JOB}"; \
	else \
	  aws logs get-log-events \
	    --log-group-name /aws/sagemaker/TrainingJobs \
	    --log-stream-name "$$LOG_STREAM" \
	    --query 'events[*].message' --output text --no-cli-pager; \
	fi

sm-cancel:
	# Stop a running job.  Usage: make sm-cancel JOB=flowmodels2-20240101-120000
ifndef JOB
	@echo "This makefile macro must be called as:"
	@echo "  make sm-cancel JOB=flowmodels2-20240101-120000  # from output of make sm-run"
	@echo
	@exit 1
endif
	@aws sagemaker stop-training-job --training-job-name ${JOB} --no-cli-pager
	@echo "Stop requested for: ${JOB}"


.PHONY: sm-what-to-do sm-create-role sm-run sm-list-jobs sm-status sm-logs sm-cancel
