# Recently-completed and next-up tasks for Flow_models
Not comprehensive long-term tracking here, just keeping organized in current work...


### Bugs List
- [ ]


### To Do
- [ ] GitHub unittests workflow
- [ ] GitHub linting workflow
- [ ] GitHub release workflow: trigger run-build on v* tags; update buildspec
      :latest condition to include v* pattern (one-line change)
- [ ] Run test runs of inference-only config
- [ ] Run test runs of awsbatch-support (overlap this with test runs above)
- [ ] Add params-file override support to AWS Batch submission
- [ ] Log AWS Batch job ID as MLflow param (equivalent of JOB_NAME for SageMaker runs)
- [ ] Refactor output_dir/model_dir to auto-derive from run ID (but note inference-mode implications)
- [ ] Use version-based ECR image tags on main branch (requires consistent update to makefile-common.mk + buildspec.yml)
- [ ] Extract sagemaker-support and awsbatch-support into standalone reusable repo

### Maybe
- [ ] Consider whether any use to logging of Dataset objects in this application
- [ ] Consider whether any use to logging the Model objects in this application
- [ ] If adding new platforms like GCP/Azure in the job-submission machinery,
      add new branches in detect_platform() and in resolve_checkpoint_dir() in
      model_training.py.
- [ ] If adding a new experiment tracking backend (eg maybe W&B), add
      elif blocks in `_start_tracking()` and in the `MLflowLoggingCallback` /
      `_build_callbacks` area above it.  Note the `if tracking_tool != "mlflow":`
      line in `_start_tracking()` - that's the block to add elif to.


### Done
- [ ]
