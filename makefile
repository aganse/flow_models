# Makefile for flow_models

export DEVICE=cpu
export PYTHON_BIN=python3.12
# Environment variables AWS_ACCT_ID and AWS_REGION are expected to exist


# This line allows the AWS Batch make commands to be run from repo root dir
include awsbatch-support/makefile-awsbatchsupport.mk

# This line allows the SageMaker Training Jobs commands to be run from repo root dir
include sagemaker-support/makefile-sagemakersupport.mk


# These are system commands used in macros below.
# (verify in a python environment to gate installation of packages)
check_venv := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo "1"; else echo "0"; fi)
# (verify in repo's root directory)
check_repo_root := $(shell if [ "$$(basename $$(pwd))" = "flow_models" ]; then echo "1"; else echo "0"; fi)
# (extract app version from setup.cfg for docker image labeling)
version := v$(shell grep -E 'current_version\s*=' setup.cfg | cut -d '=' -f2 | tr -d ' ')

create-env:
ifeq ($(check_repo_root), 1)
	@next_venv=$$($(PYTHON_BIN) -c "import os; max_val = max([int(d.replace('.venv', '')) for d in os.listdir('.') if d.startswith('.venv') and d.replace('.venv', '').isdigit()] + [0]); print(f'.venv{max_val+1}')"); \
	echo "Creating/installing new python env ${PWD}/$$next_venv"; \
	bash -c "$(PYTHON_BIN) -m venv $$next_venv && source $${next_venv}/bin/activate && pip install -r requirements.txt"
else
	@echo "Not in root directory of flow_models repo."
endif

install-dev:
ifeq ($(check_venv), 1)
	@echo "Installing dev packages with pip..."
	pip install -r requirements-dev.txt
else
	@echo "Not in a python virtual environment. Skipping pip install of dev packages."
endif

unittests:
	python -m unittest -v

lint:
	flake8 .

build-cpu:
	# Build CPU image for local dev/testing.
	docker build -t $(ECR_REPO):$(version)-cpu .

build-gpu:
	# Build GPU image for cloud training (best done via make run-build on AWS CodeBuild).
	docker build -t ${ECR_REPO}:$(version)-gpu .

run-local:
	# Run a training script locally in Docker.
	# Usage: make run-local SCRIPT=1  (or SCRIPT=2, etc.)
ifndef SCRIPT
	@echo "Usage: make run-local SCRIPT=1  (or SCRIPT=2, etc.)"
	@exit 1
endif
	$(eval DATA_MOUNT := $(shell \
	  if echo "$(IMAGES_PATH)" | grep -q "^s3://"; \
	  then echo ""; \
	  else echo "-v $(IMAGES_PATH):$(IMAGES_PATH)"; \
	  fi))
	docker run --rm -it \
	  -e TRAINING_SCRIPT=$(SCRIPT) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	  -e IMAGES_PATH=$(IMAGES_PATH) \
	  -e WEIGHTS_PATH=$(WEIGHTS_PATH) \
	  $(DATA_MOUNT) \
	  -v $(PWD)/params:/opt/ml/input/data/params \
	  $(ECR_REPO):$(version)-$(DEVICE)


# ensures all entries run every time since these aren't files
.PHONY: create-env install-dev unittests lint build-cpu build-gpu run-local
