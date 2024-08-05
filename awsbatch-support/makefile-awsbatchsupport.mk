# Note: environment variables AWS_ACCT_ID and AWS_REGION are expected to exist
# in the environment before calling these makefile macros.
#
# Note: two variables are purposely exported to the environment (to be env vars)
# whereas the rest are just within-makefile-vars, because these two are passed
# to substitute into a file.  So be careful to not arbitrarily remove the
# "export" on those two variables below (JOB_DEF_NAME and ECR_REPO_URI).


# You can set these vars to tailor to your naming preferences:
ECR_REPO=flow_models
CODEBUILD_PROJ=flow_models_build
COMPUTE_ENV_NAME=GPUcompenv
JOB_QUEUE_NAME=GPUJobQueue
export JOB_DEF_NAME=GPUJobDefinition
DEVICE=gpu  # cpu or gpu


# these vars are used in commands in the make macros down below:
export ECR_REPO_URI=${AWS_ACCT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}
NETWORKING=subnets=${AWSBATCH_SUBNET},securityGroupIds=${AWSBATCH_SG}
INSTANCE_ROLE=arn:aws:iam::${AWS_ACCT_ID}:instance-profile/BatchInstanceProfile
ROLES=instanceRole=${INSTANCE_ROLE}
EXTRA_ARGS=type=EC2,${NETWORKING},${ROLES},tags={Name=AWSBatchInstance}

what-to-do:
	# Just a quick summary of available makefile macros, group by section:
	@echo "once/rarely:    create-ecr-repo create-codebuild-role create-batch-instance-profile"
	@echo "sometimes:      create-codebuild-project run-build"
	@echo "sometimes:      create-compute-env create-job-queue register-job-definition"
	@echo "more often:     run-batchjob"
	@echo "image checks:   list-ecr-repos"
	@echo "compute checks: list-compute-resources delete-compute-resources1 delete-compute-resources2"
	@echo "job checks:     list-job-status JOBID=12345678"

create-ecr-repo:
	# Create the repo in ECR where this app's docker images will be held on AWS
	# (one-time/rare run)
	@aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

check-codebuild-role-exists:
	# Supports create-codebuild-role with a verification check
	@aws iam get-role --role-name CodeBuildServiceRole > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Role CodeBuildServiceRole already exists. Skipping creation."; \
		exit 1; \
	fi

create-codebuild-role: check-codebuild-role-exists
	# Generates CodeBuildServiceRole needed for create-codebuild-project.
	# (one-time/rare run)
	@echo "Creating role CodeBuildServiceRole..."
	@aws iam create-role \
		--role-name CodeBuildServiceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/codebuild-trust-policy.json
	@echo "Role CodeBuildServiceRole created successfully."
	@aws iam create-policy --policy-name CodeBuildCloudWatchPolicy \
		--policy-document file://awsbatch-support/codebuild-cloudwatch-policy.json
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

create-batch-executation-role:
	@aws iam create-role --role-name BatchExecutionRole --assume-role-policy-document file://awsbatch-support/batch-trust-policy.json
	@aws iam put-role-policy --role-name BatchExecutionRole --policy-name BatchExecutionPermissions --policy-document file://awsbatch-support/batch-permissions-policy.json





create-batch-instance-profile:
	# Support create-compute-env by supplying a BatchInstanceProfile
	# (one-time/rare run)
	@aws iam create-role --role-name BatchInstanceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/instance-trust-policy.json
	@aws iam attach-role-policy --role-name BatchInstanceRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
	@aws iam create-instance-profile --instance-profile-name BatchInstanceProfile
	@aws iam add-role-to-instance-profile --instance-profile-name BatchInstanceProfile \
		--role-name BatchInstanceRole
	#@aws iam get-instance-profile --instance-profile-name BatchInstanceProfile

create-codebuild-project:
	# Set up the process to get code from Github branch, build docker image, and push to ECR repo.
	# (occasional run - to push new training code/image)
	@aws codebuild create-project \
	    --name ${CODEBUILD_PROJ} \
		--source "type=GITHUB,location=https://github.com/aganse/flow_models.git,buildspec=awsbatch-support/buildspec.yml" \
	    --artifacts "type=NO_ARTIFACTS" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL,environmentVariables=[{name='ECR_REPO_URI', value='${ECR_REPO_URI}'},{name='DEVICE', value='${DEVICE}'}]" \
		--service-role arn:aws:iam::${AWS_ACCT_ID}:role/CodeBuildServiceRole
	    # (if needed auth later, we could append ",auth={type=OAUTH,resource=token}" to --source arg)

run-build:
	# Actually run the process to get code from Github branch, build docker image, and push to ECR repo.
	# (occasional run - to push new training code/image)
	@aws codebuild start-build --project-name ${CODEBUILD_PROJ}
	# to check build status in cli:
	# aws codebuild batch-get-builds --ids <arn:etc.etc.etc from start-build output, or from console>



# Assumes pre-existing AWS subnet within your VPC, and a simple security group
# with no inbound rules and default outbound 0.0.0.0; name this group something
# like aws-batch-instance, add to the inbound rules of your mlflow instance,
# and set env vars ${AWSBATCH_SUBNET} and ${AWSBATCH_SG} in your env so this
# script can use them below.
# To create sg via cli instead of manually in aws dashboard:
# aws ec2 create-security-group --group-name aws-batch-instance --description "Security group for AWS Batch compute" --vpc-id your-vpc-id
create-compute-env:
	# Create batch compute environment - g4dn.xlarge have 4 vCpus so pinning vCpus to 4.
	# (occasional run - for each set of batch runs)
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
	# Create batch job queue.
	# (occasional run - for each set of batch runs)
	@aws batch create-job-queue --job-queue-name ${JOB_QUEUE_NAME} \
		--compute-environment-order order=1,computeEnvironment=${COMPUTE_ENV_NAME} \
		--priority 1 --no-cli-pager

register-job-definition:
	# Register batch job definition.
	# (occasional run - for each set of batch runs)
	# The aws command requires an ECR_REPO_URI in this json file which varies
	# per user, so before submitting json file, we substitute in the $ECR_REPO_URI
	# environment variable below:
	@envsubst < awsbatch-support/job_definition_template.json > /tmp/job-definition.json \
	&& aws batch register-job-definition --cli-input-json file:///tmp/job-definition.json --no-cli-pager
	rm /tmp/job-definition.json

run-batchjob:
	# Run the batch job in the container from ECR, on an AWS remote GPU instance.
	@aws batch submit-job --job-name MyGPUJob --job-queue ${JOB_QUEUE_NAME} --job-definition ${JOB_DEF_NAME} --no-cli-pager
	# Command returns immediately and provides a job-id, to enter for check-job-status and cancel-job macros.
	# Job log output is found in the AWS CloudWatch Console:
	#     https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2




list-ecr-repos:
	# List the repos in ECR to confirm new one is there after create-ecr-repo.
	@aws ecr describe-repositories --query 'repositories[*].repositoryName' --output text | \
	while read repo; do echo "ECR Repository $${repo}:"; aws ecr list-images --repository-name "$${repo}" --query 'imageIds[*]' --output text --no-cli-pager; done

list-roles:
	# List the policies attached to CodeBuildServiceRole and AWSBatchServiceRole.
	@-aws iam list-attached-role-policies --role-name CodeBuildServiceRole --no-cli-pager
	@-aws iam list-attached-role-policies --role-name AWSBatchServiceRole --no-cli-pager

delete-roles:
	# Before we can delete the roles, we must detach their policies from them:
	@-aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
	@-aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess"
	@-aws iam detach-role-policy --role-name AWSBatchServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSBatchServiceRolePolicy"
	# Deleting CodeBuildServiceRole and AWSBatchServiceRole
	@-aws iam delete-role --role-name CodeBuildServiceRole
	@-aws iam delete-role --role-name AWSBatchServiceRole

list-jobs:
	@echo "posixtime	status	statusReason		jobName		jobID"
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status SUBMITTED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status PENDING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status RUNNABLE --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status STARTING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status RUNNING --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status SUCCEEDED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager
	@-aws batch list-jobs --job-queue ${JOB_QUEUE_NAME} --job-status FAILED --query 'jobSummaryList[*].[createdAt,status,statusReason,jobName,jobId]' --output text --no-cli-pager

list-compute-resources:
	# List out the AWS Batch related resources (compute-envs, job-queues, job-defs, auto-scale-groups, ec2-instances).
	@-aws batch describe-compute-environments --query 'computeEnvironments[*].[status, computeEnvironmentName, statusReason]' --output json | jq -r '["compute-env"] + .[] | @tsv'
	@-aws batch describe-job-definitions --query 'jobDefinitions[*].[status, jobDefinitionName, revision]' --output json | jq -r '.[] | ["job-definition"] + . | @tsv'
	@-aws autoscaling describe-auto-scaling-groups --query 'AutoScalingGroups[*].[Status, HealthStatus, AutoScalingGroupName]' --output json | jq -r '["autoscale-group"] + .[] | @tsv'
	@-aws batch describe-job-queues --query 'jobQueues[*].[state, jobQueueName]' --output json | jq -r '["job-queue"] + .[] | @tsv'
	@-aws ec2 describe-instances --query 'Reservations[*].Instances[*].[State.Name, InstanceId, LaunchTime, Tags[?Key==`Name`].Value | [0]]' --output json | jq -r '.[] | ["ec2-instance"] + .[] | @tsv'

delete-compute-resources1:
	# Wait a few minutes after running this macro, then run delete-compute-resources2.
	# I.e. the update-compute-environment to DISABLED must go thru before the delete.
	@-aws batch update-job-queue --job-queue ${JOB_QUEUE_NAME} --state DISABLED
	@-aws batch delete-job-queue --job-queue ${JOB_QUEUE_NAME} --no-cli-pager
	@REVISIONS=$(aws batch describe-job-definitions --job-definition-name $JOB_DEF_NAME --status "ACTIVE" --query 'jobDefinitions[*].revision' --output text)
	@for VERSION in ${REVISIONS}; do aws batch deregister-job-definition --job-definition "${JOB_DEF_NAME}:${VERSION}"; done
	@-aws batch update-compute-environment --compute-environment ${COMPUTE_ENV_NAME} --state DISABLED --no-cli-pager

delete-compute-resources2:
	# Wait for the state of update-compute-resources1 to settle first, then run this:
	@-aws batch delete-compute-environment --compute-environment ${COMPUTE_ENV_NAME} --no-cli-pager

list-job-status:
	# List status of a run-batch job that's still in progress, based on JOBID from run-batch
ifndef JOBID
	@echo "This makefile macro must be called as:"                                       
	@echo "  make check-job-status JOBID=12345678  # comes from output of `make run-batch`"
	@echo                                                                                
endif                                                                                    
	@aws batch describe-jobs --jobs $${JOBID} --no-cli-pager --output text

cancel-job:
	# Cancel a run-batch job that's still in progress, based on JOBID from run-batch.
ifndef JOBID
	@echo "This makefile macro must be called as:"                                       
	@echo "  make cancel-job JOBID=12345678  # comes from output of `make run-batch`"
	@echo                                                                                
endif                                                                                    
	@aws batch cancel-job --job-id $${JOBID} --reason "Cancelling job"


push-to-ecr:
	# Push docker image to AWS ECR from local environment.
	# eg could build it on g4dn.xlarge instance and push to ECR, then after that
	# could run image in AWS Batch for future runs.
	# (Not used when using codebuild steps.)
ifndef DEVICE
	@echo "This makefile macro must be called as:"                                       
	@echo "  make push-to-ecr DEVICE=gpu   # or DEVICE=cpu"
	@echo "The DEVICE determines whether the CPU or GPU version of the app image gets pushed to ECR."
	@echo                                                                                
endif                                                                                    
	@docker tag ${ECR_REPO}:${version}-$${DEVICE} ${ECR_REPO_URI}:latest  # tag the cpu or gpu image as 'latest'
	@aws ecr get-login-password --region $${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO_URI}  # login to ECR
	@docker push ${ECR_REPO_URI}:latest  # push the image


# This listing ensures all entries run every time since these aren't files.
.PHONY: create-ecr-repo check-codebuild-role-exists create-codebuild-role \
	create-batch-instance-profile create-codebuild-project run-build \
	create-compute-env create-job-queue register-job-definition run-batch \
	list-ecr-repos list-roles delete-roles \
	list-compute-resources delete-compute-resources1 delete-compute-resources2 \
	check-job-status cancel-job push-to-ecr
