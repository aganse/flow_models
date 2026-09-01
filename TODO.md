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
- [x] Log original images_path S3 URI to MLflow (not in-container path)
- [x] Log submitting user to MLflow user field across all run paths


### Broader Notes
  This is all on the theme of separating off the AWS/cloud job submission
  toolset as a separate repo/tool from the modeling. There still remain some
  to-dos to prep for that, discussing separately here for planning.

  Well-separated — no concern:
  * file_utils.py — boto3/S3 is used for data loading (image pipeline), not job
    submission. That's a genuine core capability useful in any context, local or cloud.
  * train_flowmodels1.py — no cloud-specific references at all.
  * train_flowmodels2.py — only line 32: os.environ.get("IMAGES_PATH", ...).
    That env var happens to be set by the makefile, but works fine locally too. No issue.

  Has entanglement — worth noting for later:
  * utils.py line 22 is the cleanest, most obvious one to flag: it reads
    /opt/ml/input/config/hyperparameters.json — that's a SageMaker-only path.
    It belongs in aws/job-support-sagemaker/, not in a general utility module.
  * flow_model.py is the bigger concern. Several things are woven into
    default_training_sequence() that are cloud/job-aware:
    - /opt/ml detection for checkpoint directory location (line 664) and Keras
      verbosity (line 756)
    - WEIGHTS_PATH S3 upload logic with boto3 (lines 613–619, 766–778)
    - MLflow tags for run_env, JOB_NAME, HOST_USER, IMAGE_TAG, IMAGES_PATH_ORIG (lines 584–604)

  The S3 upload and /opt/ml checkpoint path are the most structurally coupled —
  they mean default_training_sequence() currently has to know about job
  infrastructure. The ideal future shape would be a thin cloud adapter layer in
  aws/ that handles pre/post (checkpoint path, S3 upload, job metadata tags) and
  calls a cloud-agnostic default_training_sequence(). That's a moderate
  refactor, but utils.py's SageMaker path is a quick win you could move whenever.
