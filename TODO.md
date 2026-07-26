# Recently-completed and next-up tasks for Flow_models
Not comprehensive long-term tracking here, just keeping organized in current work...

### TODO
- [ ] Run test runs re improving blurry simulated images
  - [ ] Tune Glow training: lower LR ~3e-5, slower decay, higher grad thresh
  - [ ] Raise num_epochs (150) and patience (40) for Glow
  - [ ] Try batch_size=32 + larger arch (steps_per_block=8, num_hidden=192)
- [ ] Run test runs of sagemaker-support (overlap this with test runs above)
- [ ] Run test runs of awsbatch-support (overlap this with test runs above)
- [ ] Add params-file override support to AWS Batch submission
- [ ] Update/improve README.md readability and organization
- [ ] GitHub unittests workflow
- [ ] GitHub linting workflow
- [ ] GitHub release workflow: trigger run-build on v* tags; update buildspec
      :latest condition to include v* pattern (one-line change)
- [ ] Extract sagemaker-support and awsbatch-support into standalone reusable repo

### Done
- [x] Add branch/device tagging to run-build; TAG arg to sm-run
- [x] Fix sim images noise (clip pixels before uint8)
- [x] Fix direct-sampling of latent space points
- [x] Add MLflow latent points convergence stats logging
- [x] Create unittest tests/test_train_step.py (11 tests, 3 classes)
- [x] Fix trainable_variables: id(v) replaces v.ref()
- [x] Harden train_step against Keras 3.x tuple wrapping
- [x] Move unwrap_batch and load_param_overrides to utils.py
- [x] Implement checkpointing + spot instance support
- [x] Fix README run instructions for params (Options 1–5)
