# Recently-completed and next-up tasks for Flow_models
Not comprehensive long-term tracking here, just keeping organized in current work...

### BUGS

### TODO
- [/] Run test runs re improving blurry simulated images
  - [/] Tune Glow training: lower LR ~3e-5, slower decay, higher grad thresh
  - [ ] Raise num_epochs (150) and patience (40) for Glow
  - [ ] Try batch_size=32 + larger arch (steps_per_block=8, num_hidden=192)
- [ ] Run test runs of awsbatch-support (overlap this with test runs above)
- [ ] Add params-file override support to AWS Batch submission
- [ ] Update/improve README.md readability and organization
- [ ] GitHub unittests workflow
- [ ] GitHub linting workflow
- [ ] GitHub release workflow: trigger run-build on v* tags; update buildspec
      :latest condition to include v* pattern (one-line change)
- [ ] Extract sagemaker-support and awsbatch-support into standalone reusable repo
- [ ] Log AWS Batch job ID as MLflow param (equivalent of JOB_NAME for SageMaker runs)
- [ ] Refactor output_dir/model_dir to auto-derive from run ID (see inference-mode implications)
- [ ] Use version-based ECR image tags on main branch (requires consistent update to makefile-common.mk + buildspec.yml)

### Done
- [x] Fix local-build/local-run/push-to-ecr image tag to use {branch}-{device} not {version}-{device}
- [X] Run test runs of sagemaker-support (overlap this with test runs above)
- [x] Implement checkpointing + spot instance support
- [x] Fix README run instructions for params (Options 1–5)
- [x] Fix sm-logs output (switch to --output json + Python for correct newlines)
- [x] Add "spot" suffix to job name when SPOT=1; show <=$ cost in sm-list
- [x] Use job-specific S3 subdirs for weights and checkpoints (fix concurrent-job collision)
- [x] Log run_env tag and job_name param to MLflow; keep only last 2 checkpoints
- [x] Delete checkpoints after final weights saved; document checkpoint_every_n_epochs
- [x] Remove validate_args from FlowModel and params files
