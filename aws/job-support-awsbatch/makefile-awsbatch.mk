# AWS Batch support makefile.
# Included from root Makefile; ECR/build targets are in job-support-common/makefile-common.mk.
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID       - AWS account ID
#   AWS_REGION        - AWS region (e.g. us-west-2)
#   AWSBATCH_SUBNET   - VPC subnet ID for Batch compute (create-compute-env)
#   AWSBATCH_SG       - Security group ID for Batch compute (create-compute-env)
#   MLFLOW_TRACKING_URI - MLflow server URI (register-job-definition)
#   IMAGES_PATH       - S3 URI of training images (register-job-definition, optional for train1)
#
# Note: several variables are purposely exported to the environment (to be
# env vars) because they are passed to envsubst to substitute into a file.
# So be careful to not arbitrarily remove the "export" on those variables.


# You can set these vars to tailor to your naming preferences:
COMPUTE_ENV_NAME=GPUcompenv
JOB_QUEUE_NAME=GPUJobQueue
export JOB_DEF_NAME=GPUJobDefinition

# Used in commands below:
NETWORKING=subnets=${AWSBATCH_SUBNET},securityGroupIds=${AWSBATCH_SG}
INSTANCE_ROLE=arn:aws:iam::${AWS_ACCT_ID}:instance-profile/BatchInstanceProfile
ROLES=instanceRole=${INSTANCE_ROLE}
EXTRA_ARGS=type=EC2,${NETWORKING},${ROLES},tags={Name=AWSBatchInstance}
# Export so envsubst can substitute them into job_definition_template.json:
export MLFLOW_TRACKING_URI
export IMAGES_PATH
export WEIGHTS_PATH

batch-help:
	@echo "once/rarely:    create-batch-executation-role  create-batch-instance-profile  create-compute-env  create-job-queue"
	@echo "sometimes:      register-job-definition"
	@echo "more often:     batch-submit"
	@echo "job checks:     batch-list  batch-status JOBID=id  batch-logs JOBID=id  batch-cancel JOBID=id"
	@echo "compute checks: list-compute-resources  delete-compute-resources1  delete-compute-resources2"
	@echo "(for ECR/CodeBuild/image targets see aws/job-support-common/makefile-common.mk)"

create-batch-executation-role:
	@aws iam create-role --role-name BatchExecutionRole --assume-role-policy-document file://aws/job-support-awsbatch/batch-trust-policy.json
	@aws iam put-role-policy --role-name BatchExecutionRole --policy-name BatchExecutionPermissions --policy-document file://aws/job-support-awsbatch/batch-permissions-policy.json

create-batch-instance-profile:
	# Support create-compute-env by supplying a BatchInstanceProfile.
	# (one-time/rare run)
	@aws iam create-role --role-name BatchInstanceRole --no-cli-pager \
		--assume-role-policy-document file://aws/job-support-awsbatch/instance-trust-policy.json
	@aws iam attach-role-policy --role-name BatchInstanceRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
	@aws iam create-instance-profile --instance-profile-name BatchInstanceProfile
	@aws iam add-role-to-instance-profile --instance-profile-name BatchInstanceProfile \
		--role-name BatchInstanceRole

create-compute-env:
	# Create Batch compute environment — g4dn.xlarge has 4 vCPUs so pinning vCPUs to 4.
	# (occasional run - for each set of batch runs)
ifndef AWSBATCH_SUBNET
	$(error AWSBATCH_SUBNET is not set. Export it before running create-compute-env)
endif
ifndef AWSBATCH_SG
	$(error AWSBATCH_SG is not set. Export it before running create-compute-env)
endif
	@aws batch create-compute-environment --compute-environment-name ${COMPUTE_ENV_NAME} --type MANAGED \
		--compute-resources instanceTypes=g4dn.xlarge,minvCpus=0,desiredvCpus=0,maxvCpus=4,${EXTRA_ARGS} \
		--service-role "" --no-cli-pager
	# Blank string gives default AWSServiceRoleForBatch (default service-linked-role).
	# Setting minvCpus=0,desiredvCpus=0 -> system terminates instances when no jobs in queue.

	# to set up to use much-cheaper spot instances later:
	# aws batch create-compute-environment --compute-environment-name ${COMPUTE_ENV_NAME} --type MANAGED \
	#	--compute-resources type=SPOT,allocationStrategy=SPOT_CAPACITY_OPTIMIZED,minvCpus=4,maxvCpus=4,desiredvCpus=4,instanceTypes=g4dn.xlarge,subnets=${AWSBATCH_SUBNET},securityGroupIds=${AWSBATCH_SG},spotIamFleetRole=arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole \
    #   --service-role ""

	@aws batch describe-compute-environments --compute-environments ${COMPUTE_ENV_NAME}

create-job-queue:
	# Create Batch job queue.
	# (occasional run - for each set of batch runs)
	@aws batch create-job-queue --job-queue-name ${JOB_QUEUE_NAME} \
		--compute-environment-order order=1,computeEnvironment=${COMPUTE_ENV_NAME} \
		--priority 1 --no-cli-pager

register-job-definition:
	# Register Batch job definition.
	# (occasional run - re-run after image or env changes)
ifndef AWS_ACCT_ID
	$(error AWS_ACCT_ID is not set. Export it before running register-job-definition)
endif
ifndef AWS_REGION
	$(error AWS_REGION is not set. Export it before running register-job-definition)
endif
ifndef MLFLOW_TRACKING_URI
	$(error MLFLOW_TRACKING_URI is not set. Export it before running register-job-definition)
endif
	@envsubst < aws/job-support-awsbatch/job_definition_template.json > /tmp/job-definition.json \
	&& aws batch register-job-definition --cli-input-json file:///tmp/job-definition.json --no-cli-pager
	rm /tmp/job-definition.json

batch-submit:
	# Submit a Batch training job.  Prints job ID immediately.
	$(eval JOB_ID := $(shell aws batch submit-job --job-name MyGPUJob \
	  --job-queue ${JOB_QUEUE_NAME} --job-definition ${JOB_DEF_NAME} \
	  --query 'jobId' --output text --no-cli-pager))
	@echo "Submitted: $(JOB_ID)"
	@echo "Check status: make batch-status JOBID=$(JOB_ID)"
	@echo "Check logs:   make batch-logs JOBID=$(JOB_ID)"

batch-list:
	# List all Batch jobs across all statuses.
	@echo "posixtime	status	statusReason		jobName		jobID"
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status SUBMITTED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status PENDING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status RUNNABLE --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status STARTING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status RUNNING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status SUCCEEDED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status FAILED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager

batch-status:
	# Show detailed status of a specific Batch job.
	# Usage: make batch-status JOBID=<id-from-batch-submit>
ifndef JOBID
	@echo "Usage: make batch-status JOBID=<id-from-batch-submit>"
	@echo
endif
	@aws batch describe-jobs --jobs $${JOBID} --no-cli-pager --output text

batch-logs:
	# Fetch CloudWatch logs for a Batch job.
	# Usage: make batch-logs JOBID=<id-from-batch-submit>
ifndef JOBID
	@echo "Usage: make batch-logs JOBID=<id-from-batch-submit>"
	@echo
	@exit 1
endif
	@LOG_STREAM=$$(aws logs describe-log-streams \
	  --log-group-name /aws/batch/job \
	  --log-stream-name-prefix test-debug/default/$(JOBID) \
	  --query 'logStreams[0].logStreamName' --output text 2>/dev/null); \
	if [ "$$LOG_STREAM" = "None" ] || [ -z "$$LOG_STREAM" ]; then \
	  echo "No log stream found yet for job $(JOBID)"; \
	else \
	  aws logs get-log-events \
	    --log-group-name /aws/batch/job \
	    --log-stream-name "$$LOG_STREAM" \
	    --query 'events[*].message' --output text --no-cli-pager; \
	fi

batch-cancel:
	# Cancel a running Batch job.
	# Usage: make batch-cancel JOBID=<id-from-batch-submit>
ifndef JOBID
	@echo "Usage: make batch-cancel JOBID=<id-from-batch-submit>"
	@echo
endif
	@aws batch cancel-job --job-id $${JOBID} --reason "Cancelling job"

list-compute-resources:
	# List compute envs, job queues, job defs, and running EC2 instances.
	@-aws batch describe-compute-environments --query 'computeEnvironments[*].[status, computeEnvironmentName, statusReason]' --output json | jq -r '["compute-env"] + .[] | @tsv'
	@-aws batch describe-job-definitions --query 'jobDefinitions[*].[status, jobDefinitionName, revision]' --output json | jq -r '.[] | ["job-definition"] + . | @tsv'
	@-aws autoscaling describe-auto-scaling-groups --query 'AutoScalingGroups[*].[Status, HealthStatus, AutoScalingGroupName]' --output json | jq -r '["autoscale-group"] + .[] | @tsv'
	@-aws batch describe-job-queues --query 'jobQueues[*].[state, jobQueueName]' --output json | jq -r '["job-queue"] + .[] | @tsv'
	@-aws ec2 describe-instances --query 'Reservations[*].Instances[*].[State.Name, InstanceId, LaunchTime, Tags[?Key==`Name`].Value | [0]]' --output json | jq -r '.[] | ["ec2-instance"] + .[] | @tsv'

delete-compute-resources1:
	# Disable and delete job queue and job definitions (step 1 of 2).
	# Wait a few minutes after this before running delete-compute-resources2.
	@-aws batch update-job-queue --job-queue ${JOB_QUEUE_NAME} --state DISABLED
	@-aws batch delete-job-queue --job-queue ${JOB_QUEUE_NAME} --no-cli-pager
	@export REVISIONS=$(aws batch describe-job-definitions --job-definition-name $JOB_DEF_NAME --status "ACTIVE" --query 'jobDefinitions[*].revision' --output text)
	@for VERSION in $${REVISIONS}; do aws batch deregister-job-definition --job-definition "${JOB_DEF_NAME}:${VERSION}"; done
	@-aws batch update-compute-environment --compute-environment ${COMPUTE_ENV_NAME} --state DISABLED --no-cli-pager

delete-compute-resources2:
	# Delete compute environment (step 2 of 2, run after step 1 settles).
	@-aws batch delete-compute-environment --compute-environment ${COMPUTE_ENV_NAME} --no-cli-pager


.PHONY: batch-help \
	create-batch-executation-role create-batch-instance-profile \
	create-compute-env create-job-queue register-job-definition \
	batch-submit batch-list batch-status batch-logs batch-cancel \
	list-compute-resources delete-compute-resources1 delete-compute-resources2
