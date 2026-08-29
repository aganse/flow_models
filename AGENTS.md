# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project centered on TensorFlow-based flow
models. Core modules live at the repo root: `flow_model.py`, `file_utils.py`,
and `utils.py`. Training entry points are `train_flowmodels1.py` and
`train_flowmodels2.py`; keep new experiment scripts in the same
`train_flowmodelsN.py` pattern. Tests live in `tests/`. Documentation and
parameter notes are in `doc/`. Generated artifacts are typically written under
`model/` and `output/`. AWS/cloud job support files (CodeBuild, ECR, SageMaker, Batch) are in `aws/`.

## Build, Test, and Development Commands
- `make create-env` — create a new numbered virtualenv (for example `.venv4`) and install runtime dependencies.
- `source .venvN/bin/activate` — activate the latest environment you created.
- `make install-dev` — install developer tools from `requirements-dev.txt` before you run the tests or linting.
- `make test` — run the test suite.
- `make lint` — run the code linting check (flake8).
- `python train_flowmodels1.py` — run a local training script; update parameters in the script first.
- `docker build --build-arg TENSORFLOW_PKG=tensorflow-cpu==2.12.0 -t flow_models:dev .` — example local CPU image build.

## Coding Style & Naming Conventions
Use 4-space indentation and follow PEP 8-style Python. Keep lines at 80
characters to match `flake8` in `setup.cfg`. Prefer descriptive `snake_case`
names for functions, variables, and files; use `CapWords` for classes. Format
with `black`, sort imports with `isort`, and verify with `flake8` before
submitting changes.

## Testing Guidelines
Tests use the standard `unittest` framework. Name new files
`tests/test_<feature>.py` and keep test methods as `test_<behavior>()`. Some
tests depend on environment-specific resources: `tests/test_file_utils.py`
expects `FLOW_MODELS_S3URI`, and `tests/test_gpu.py` assumes TensorFlow can
access a GPU. For routine changes, run targeted tests first, then `python -m
unittest -v`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Fix simpt sampling` or
`Add prefetching pipeline for training`. Follow that style: one-line summary,
present tense, under ~72 characters. Pull requests should explain the training
or utility change, note any config/env requirements, and include sample output
paths or screenshots for plots when behavior changes. Avoid committing large
generated files unless they are intentional repository fixtures.
