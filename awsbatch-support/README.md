
currently have almost successfully finished up thru creating compute envinronment.
still having some trouble with remaking compute environment - seems that lags
between entering aws cli commands are coming into play.

and after that, `make create-job-queue` has thrown an error that compute env
is "not valid".  so presumably once the above is resolved with compute env,
perhaps create-job-queue will be alright.



build-and-push-local-image: build-gpu push-to-ecr

define-ecr-repo: create-ecr-repo list-ecr-repos  # very rarely

define-roles: create-codebuild-role create-batch-role list-roles  # very rarely

define-the-compute: create-compute-env create-job-queue register-job-definition  # rarely

build-image-and-run-job: create-project run-batch  # regularly

Occasional commands to run manually:
check-job-status, cancel-job
list-roles
delete-roles
