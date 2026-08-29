# SageMaker Training Jobs support makefile.
# Included from root Makefile; ECR/build targets are in job-support-common/makefile-common.mk.
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID   - AWS account ID
#   AWS_REGION    - AWS region (e.g. us-west-2)
#   SM_SUBNET     - VPC subnet ID for SageMaker compute
#   SM_SG         - Security group ID for SageMaker compute
#   IMAGES_PATH   - S3 URI of training data (e.g. s3://mybucket/afhq)
#   WEIGHTS_PATH  - S3 URI for model weights persistence (e.g. s3://mybucket/weights, optional)
#   MLFLOW_TRACKING_URI - MLflow server URI (e.g. http://10.0.1.50:5000)


# Repo-specific, non-sensitive settings — adjust to taste:
SM_INSTANCE_TYPE=ml.g4dn.xlarge
SM_VOLUME_SIZE=50
SM_MAX_RUNTIME=86400
SM_MAX_WAIT ?= 90000
SM_JOB_PREFIX=flowmodels
TAG ?= latest  # ECR image tag to use; override with e.g. make sm-submit SCRIPT=2 TAG=cleanup-gpu

# Spot instance flags: set when SPOT=1 is passed on the command line.
# SM_MAX_WAIT is the total wall-clock deadline including interruption waits
# (must be >= SM_MAX_RUNTIME); override with e.g. make sm-submit SPOT=1 SM_MAX_WAIT=36000
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


sm-help:
	@echo "once/rarely:   sm-create-sg VPC_ID=vpc-xxx,  sm-create-role,  sm-list-role"
	@echo "sometimes:     build  (shared CodeBuild pipeline, pushes to ECR)"
	@echo "image checks:  ecr-list-images"
	@echo "more often:    sm-submit SCRIPT=1 [TAG=main-gpu]  (TAG defaults to latest)"
	@echo "job checks:    sm-list,  sm-status JOB=name,  sm-logs JOB=name,  sm-cancel JOB=name"

sm-create-sg:
	# Create a SageMaker security group (one-time/rare run).
	# Allow-all outbound (AWS default), no inbound. Name encodes creation date.
	# Usage: make sm-create-sg VPC_ID=vpc-xxxxxxxx
	# After running: export SM_SG=<id printed below>, and add an inbound rule on
	# your EC2's SG allowing port 5000 (MLflow) from this new SG.
ifndef VPC_ID
	@echo "Usage: make sm-create-sg VPC_ID=vpc-xxxxxxxx"
	@exit 1
endif
	$(eval SG_NAME := sagemaker-to-mlflow-$(shell date +%Y-%m-%d))
	$(eval SG_ID := $(shell aws ec2 create-security-group \
	  --group-name $(SG_NAME) \
	  --description "SageMaker training jobs outbound to MLflow EC2" \
	  --vpc-id $(VPC_ID) \
	  --query 'GroupId' --output text --no-cli-pager))
	@aws ec2 create-tags --resources $(SG_ID) \
	  --tags Key=Name,Value=$(SG_NAME) --no-cli-pager
	@echo "Created: $(SG_ID)  ($(SG_NAME))"
	@echo "Outbound: allow-all (AWS default). Inbound: none."
	@echo "Next steps:"
	@echo "  export SM_SG=$(SG_ID)"
	@echo "  Add inbound rule on your EC2's SG: port 5000 from $(SG_ID)"

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

sm-list-role:
	# Show whether SageMakerExecutionRole exists and its ARN/creation date.
	@aws iam get-role --role-name SageMakerExecutionRole --no-cli-pager \
	  --query '{RoleName:Role.RoleName,Created:Role.CreateDate,Arn:Role.Arn}' \
	  --output table 2>/dev/null || echo "SageMakerExecutionRole does not exist."

sm-submit:
	# Submit a SageMaker Training Job for train_flowmodelsN.py.
	# Usage: make sm-submit SCRIPT=1  (or SCRIPT=2, etc.)
ifndef SCRIPT
	@echo "Usage: make sm-submit SCRIPT=1  # or SCRIPT=2, etc."
	@echo
	@exit 1
endif
ifndef MLFLOW_TRACKING_URI
	$(error MLFLOW_TRACKING_URI is not set. You must export it before running sm-submit)
endif
	@if [ "$(SCRIPT)" != "1" ] && ! echo "$(IMAGES_PATH)" | grep -q "^s3://"; then \
	  echo "Error: IMAGES_PATH must be an s3:// URI for SCRIPT=$(SCRIPT) SageMaker runs with images" >&2; exit 1; fi
	@if [ -n "$(WEIGHTS_PATH)" ] && ! echo "$(WEIGHTS_PATH)" | grep -q "^s3://"; then \
	  echo "Error: WEIGHTS_PATH must be an s3:// URI if set, got: '$(WEIGHTS_PATH)'" >&2; exit 1; fi
ifndef AWS_ACCT_ID
	$(error AWS_ACCT_ID is not set. Export it before running sm-submit)
endif
ifndef AWS_REGION
	$(error AWS_REGION is not set. Export it before running sm-submit)
endif
ifndef SM_SUBNET
	$(error SM_SUBNET is not set. Export it before running sm-submit)
endif
ifndef SM_SG
	$(error SM_SG is not set. Export it before running sm-submit)
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

sm-list:
	# List recent SageMaker training jobs and their status.
	@aws sagemaker list-training-jobs \
	  --sort-by CreationTime --sort-order Descending \
	  --max-results 20 \
	  --query 'TrainingJobSummaries[*].[CreationTime,TrainingJobStatus,TrainingJobName]' \
	  --output text --no-cli-pager

sm-status:
	# Show status of a submitted job.  Usage: make sm-status JOB=flowmodels2-20240101-120000
ifndef JOB
	@echo "Usage: make sm-status JOB=flowmodels2-20240101-120000  # from output of make sm-submit"
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
	@echo "Usage: make sm-logs JOB=flowmodels2-20240101-120000  # from output of make sm-submit"
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
	@echo "Usage: make sm-cancel JOB=flowmodels2-20240101-120000  # from output of make sm-submit"
	@echo
	@exit 1
endif
	@aws sagemaker stop-training-job --training-job-name ${JOB} --no-cli-pager
	@echo "Stop requested for: ${JOB}"


.PHONY: sm-help sm-create-sg sm-create-role sm-list-role sm-submit sm-list sm-status sm-logs sm-cancel
