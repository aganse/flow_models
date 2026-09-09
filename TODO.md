# Recently-completed and next-up tasks for Flow_models
Not comprehensive long-term tracking here, just keeping organized in current work...


### Bugs List
- [ ]


### To Do
- [/] Run test runs re improving blurry simulated images
  - [/] Tune Glow training: lower LR ~3e-5, slower decay, higher grad thresh
  - [ ] Raise num_epochs (150) and patience (40) for Glow
  - [ ] Try batch_size=32 + larger arch (steps_per_block=8, num_hidden=192)
- [ ] GitHub unittests workflow
- [ ] GitHub linting workflow
- [ ] GitHub release workflow: trigger run-build on v* tags; update buildspec
      :latest condition to include v* pattern (one-line change)
- [ ] Run test runs of awsbatch-support (overlap this with test runs above)
- [ ] Add params-file override support to AWS Batch submission
- [ ] Log AWS Batch job ID as MLflow param (equivalent of JOB_NAME for SageMaker runs)
- [ ] Refactor output_dir/model_dir to auto-derive from run ID (but note inference-mode implications)
- [ ] Use version-based ECR image tags on main branch (requires consistent update to makefile-common.mk + buildspec.yml)
- [ ] Extract sagemaker-support and awsbatch-support into standalone reusable repo
- [ ] Consider whether any use to logging of Dataset objects in this application
- [ ] Consider whether any use to logging the Model objects in this application


### Done
- [ ]
