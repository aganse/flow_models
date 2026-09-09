"""Training orchestration for flow models.

This module is intentionally decoupled from any specific cloud platform or
experiment-tracking backend.  Platform detection (detect_platform) and
checkpoint-directory resolution (resolve_checkpoint_dir) are the only points
of platform awareness; all other code treats the resulting strings as opaque
configuration.

Typical usage:
    from file_utils import get_data_generator
    from model_training import default_training_sequence, load_param_overrides

    load_param_overrides(run_params, training_params, model_arch_params)
    train_gen = get_data_generator(...)
    flow_model, history = default_training_sequence(
        train_gen, run_params, training_params, model_arch_params
    )
"""

import base64
import getpass
import glob
import io
import json
import os
import re
import sys
from datetime import datetime

import mlflow
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from file_utils import infinite_generator
from flow_model import FlowModel

_RUN_ID_FILENAME = "mlflow_run_id.txt"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform():
    """Return the current execution platform as a string.

    Returns one of: 'sagemaker', 'awsbatch', 'local'.
    Add new branches here (and in resolve_checkpoint_dir) when adding platforms.
    """
    if os.path.exists("/opt/ml"):
        return "sagemaker"
    if os.environ.get("AWS_BATCH_JOB_ID"):
        return "awsbatch"
    return "local"


def resolve_checkpoint_dir(run_params, platform):
    """Return the appropriate checkpoint directory path for the given platform."""
    if platform == "sagemaker":
        return "/opt/ml/checkpoints"
    return os.path.join(run_params["model_dir"], "checkpoints")


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def load_param_overrides(
    run_params,
    training_params,
    model_arch_params,
    hp_path="/opt/ml/input/config/hyperparameters.json",
):
    """Merge base64-encoded JSON parameter overrides into the three param dicts.

    No-op when the file does not exist (local runs without Docker).  The
    default path is the SageMaker hyperparameter file written by the
    entrypoint; pass a different hp_path for other platforms.
    """
    if not os.path.exists(hp_path):
        return
    with open(hp_path) as f:
        hparams = json.load(f)
    if "params" in hparams:
        overrides = json.loads(base64.b64decode(hparams["params"]))
        run_params.update(overrides.get("run_params", {}))
        training_params.update(overrides.get("training_params", {}))
        model_arch_params.update(overrides.get("model_arch_params", {}))


# ---------------------------------------------------------------------------
# MLflow training callback
# ---------------------------------------------------------------------------

class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    """Mirror Keras epoch-end metrics into the active MLflow run."""

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        metrics = {}
        for key, value in logs.items():
            if value is None:
                continue
            if np.isscalar(value):
                metrics[key] = float(value)
            else:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
        if metrics:
            mlflow.log_metrics(metrics, step=epoch)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten_params(params):
    flat_params = {}
    if not params:
        return flat_params
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            flat_params[key] = value
        else:
            flat_params[key] = repr(value)
    return flat_params


def _capture_and_save_summary(model, image_shape, output_dir):
    """Generate model summary text, save to file, and return the path."""
    try:
        dummy_input = tf.zeros((1, *image_shape), dtype=tf.float32)
        _ = model(dummy_input)
    except Exception:
        pass

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_text = stream.getvalue()

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "flow_model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(summary_text, end="")
    return summary_path


def _build_optimizer(training_params, model_arch_params):
    """Return an (optimizer, jit_compile) pair for the given configuration."""
    lr = training_params["learning_rate"]
    if isinstance(lr, float):
        lrate = lr
    elif isinstance(lr, list) and len(lr) == 3:
        lrate = ExponentialDecay(
            lr[0], decay_steps=lr[1], decay_rate=lr[2], staircase=True
        )
    else:
        print("model_training: error: learning_rate must be a float or list of length 3.")
        sys.exit(1)

    jit_compile = training_params["jit_compile"]
    if model_arch_params.get("bijector") == "glow" and jit_compile:
        # Glow's trainable 1x1 conv creates permutation variables on CPU;
        # XLA cannot access CPU variables from GPU kernels. Also, TFP 0.25
        # Glow has int32/int64 shape issues that only fully surface under XLA.
        print(
            "Note: jit_compile disabled for Glow bijector (XLA/TFP incompatibility).\n",
            flush=True,
        )
        jit_compile = False

    return Adam(learning_rate=lrate), jit_compile


def _load_checkpoint(flow_model, ckpt_dir):
    """Load the latest checkpoint if one exists; return the starting epoch."""
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt-*.weights.h5")))
    if not ckpt_files:
        return 0
    latest = ckpt_files[-1]
    initial_epoch = int(re.search(r"ckpt-(\d+)", latest).group(1))
    flow_model.load_weights(latest)
    print(f"Resuming from checkpoint epoch {initial_epoch}: {latest}\n", flush=True)
    return initial_epoch


def _build_callbacks(training_params, tracking_tool, ckpt_dir):
    """Build and return the list of Keras training callbacks."""
    callbacks = []
    tracking_port = training_params.get("tracking_port")

    if training_params["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStopping(
                monitor="loss",
                patience=training_params["early_stopping_patience"],
                restore_best_weights=True,
            )
        )

    if tracking_tool == "tensorboard":
        log_dir = f"./logs/train/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        callbacks.append(
            TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
        )
        if tracking_port:
            print(
                f"TensorBoard logs: {log_dir} "
                f"(launch via `tensorboard --logdir {log_dir} --port {tracking_port}`)"
            )
        else:
            print(f"TensorBoard logs: {log_dir}")
    elif tracking_tool == "mlflow":
        callbacks.append(MLflowLoggingCallback())

    if training_params.get("checkpoint_every_n_epochs", 0) > 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        steps_per_epoch = (
            training_params["num_data_input"]
            // training_params["batch_size"]
            * training_params["augmentation_factor"]
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(ckpt_dir, "ckpt-{epoch:04d}.weights.h5"),
                save_weights_only=True,
                save_freq=training_params["checkpoint_every_n_epochs"] * steps_per_epoch,
                verbose=1,
            )
        )

        class _KeepLast2Checkpoints(tf.keras.callbacks.Callback):
            def __init__(self, ckpt_dir):
                super().__init__()
                self._ckpt_dir = ckpt_dir

            def on_epoch_end(self, epoch, logs=None):
                files = sorted(
                    glob.glob(os.path.join(self._ckpt_dir, "ckpt-*.weights.h5"))
                )
                for f in files[:-2]:
                    os.remove(f)

        callbacks.append(_KeepLast2Checkpoints(ckpt_dir))

    return callbacks


def _validate_weights_dest(training_params):
    """Fail fast at job start if the configured S3 weights destination is unreachable."""
    if not training_params.get("save_model_weights"):
        return
    weights_dest = os.environ.get("WEIGHTS_PATH", "")
    if weights_dest.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        parsed = urlparse(weights_dest)
        try:
            boto3.client("s3").head_bucket(Bucket=parsed.netloc)
        except Exception as e:
            print(
                f"FATAL: cannot access S3 bucket {parsed.netloc!r} "
                f"for model weights upload: {e}",
                flush=True,
            )
            sys.exit(1)


def _save_training_outputs(
    flow_model, run_params, training_params, model_arch_params, ckpt_dir, tracking_active
):
    """Save model weights, clean up checkpoints, and write summary/arch files."""
    os.makedirs(run_params["model_dir"], exist_ok=True)
    _job_name = os.environ.get("JOB_NAME", "")

    if training_params["save_model_weights"]:
        weights_path = os.path.join(
            run_params["model_dir"], "model_weights.weights.h5"
        )
        flow_model.save_weights(weights_path)
        print("Model weights saved to file.\n", flush=True)

        weights_dest = os.environ.get("WEIGHTS_PATH")
        if weights_dest:
            if weights_dest.startswith("s3://"):
                import boto3
                from urllib.parse import urlparse
                parsed = urlparse(weights_dest)
                bucket = parsed.netloc
                key_prefix = parsed.path.lstrip("/")
                if _job_name:
                    key_prefix = f"{key_prefix}/{_job_name}"
                key = f"{key_prefix}/{os.path.basename(weights_path)}"
                boto3.client("s3").upload_file(weights_path, bucket, key)
                print(
                    f"Model weights uploaded to s3://{bucket}/{key}\n", flush=True
                )
            else:
                import shutil
                os.makedirs(weights_dest, exist_ok=True)
                shutil.copy2(weights_path, weights_dest)
                print(f"Model weights copied to {weights_dest}\n", flush=True)

        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            for f in glob.glob(os.path.join(ckpt_dir, "ckpt-*.weights.h5")):
                os.remove(f)
            run_id_file = os.path.join(ckpt_dir, _RUN_ID_FILENAME)
            if os.path.exists(run_id_file):
                os.remove(run_id_file)
            print("Checkpoints deleted (final weights saved).\n", flush=True)

    summary_path = _capture_and_save_summary(
        flow_model,
        model_arch_params["image_shape"],
        run_params["model_dir"],
    )
    arch_path = os.path.join(run_params["model_dir"], "model_arch.json")
    with open(arch_path, "w") as f:
        f.write(flow_model.to_json())
    if tracking_active:
        run_params["model_summary_path"] = summary_path
        run_params["mlflow_run_open"] = True


def _start_tracking(run_params, training_params, model_arch_params, ckpt_dir, platform):
    """Set up experiment tracking and start (or resume) a run.

    Returns True if an MLflow run was started, False otherwise.
    Add new tracking backends with elif blocks in this function and in the
    MLflowLoggingCallback / _build_callbacks area above.

    MLflow run continuation across spot-instance restarts:
    The checkpoint directory is synced to/from S3 by SageMaker's managed spot
    training between instance replacements.  When checkpointing is enabled we
    persist the run_id there so a restarted instance can resume logging into
    the same MLflow run.  The file is removed after successful job completion
    (in _save_training_outputs) so subsequent independent runs start fresh.
    """
    tracking_tool = training_params.get("tracking_tool")
    valid_tools = {None, "tensorboard", "mlflow"}
    if tracking_tool not in valid_tools:
        raise ValueError(
            f"Unsupported tracking_tool {tracking_tool!r}. "
            f"Expected one of {valid_tools - {None}} or None."
        )

    if tracking_tool != "mlflow":
        # Future backends: elif tracking_tool == "wandb": ...
        return False

    tracking_port = training_params.get("tracking_port")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif tracking_port:
        mlflow.set_tracking_uri(f"http://localhost:{tracking_port}")

    experiment_name = training_params.get(
        "tracking_expt_name", run_params.get("dataset", "flow_model_training")
    )
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(
            f"FATAL: cannot reach MLflow server at "
            f"{mlflow.get_tracking_uri()!r}: {e}",
            flush=True,
        )
        sys.exit(1)

    if mlflow.active_run():
        mlflow.end_run()

    # Check for a persisted run_id from a previous spot-instance attempt.
    # Only relevant (and only written) when checkpointing is enabled, since
    # SageMaker only syncs /opt/ml/checkpoints/ when checkpoint-config is set.
    checkpoint_enabled = training_params.get("checkpoint_every_n_epochs", 0) > 0
    run_id_file = os.path.join(ckpt_dir, _RUN_ID_FILENAME)
    existing_run_id = None
    if checkpoint_enabled and os.path.exists(run_id_file):
        with open(run_id_file) as f:
            existing_run_id = f.read().strip() or None
        if existing_run_id:
            print(
                f"Spot restart detected: resuming MLflow run {existing_run_id}.\n",
                flush=True,
            )

    if existing_run_id:
        try:
            mlflow.start_run(run_id=existing_run_id, log_system_metrics=True)
        except Exception:
            # Run was deleted or is otherwise unavailable; fall through to a fresh run.
            existing_run_id = None

    if not existing_run_id:
        dataset = run_params.get("dataset", "flow_model_run")
        num_gen = run_params.get("num_gen_sims", "NA")
        mlflow.start_run(run_name=f"{dataset}_{num_gen}", log_system_metrics=True)

    # Tags are idempotent — safe to re-set when resuming a run.
    mlflow.set_tag("mlflow.user", os.environ.get("HOST_USER", getpass.getuser()))
    mlflow.set_tag("image_tag", os.environ.get("IMAGE_TAG", "[local]"))
    mlflow.set_tag("run_env", platform)
    _job_name = os.environ.get("JOB_NAME", "")
    if _job_name:
        mlflow.log_param("job_name", _job_name)

    # Log params only on a fresh run; MLflow raises if you try to overwrite
    # an existing param with a different value.
    if not existing_run_id:
        params_for_logging = {
            **run_params,
            **training_params,
            **model_arch_params,
            "tracking_tool": tracking_tool,
        }
        params_for_logging.pop("output_dir", None)
        _images_path_orig = os.environ.get("IMAGES_PATH_ORIG", "")
        if _images_path_orig:
            params_for_logging["images_path"] = _images_path_orig
        mlflow.log_params(_flatten_params(params_for_logging))

    # Persist run_id so a spot restart can resume this run.
    if checkpoint_enabled:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(run_id_file, "w") as f:
            f.write(mlflow.active_run().info.run_id)

    run_params["mlflow_run_id"] = mlflow.active_run().info.run_id
    return True


def _make_dataset(train_gen, model_arch_params):
    """Wrap an infinite data generator in a prefetched tf.data.Dataset."""
    def _gen():
        for batch in infinite_generator(train_gen):
            if isinstance(batch, (tuple, list)):
                yield batch[0]
            else:
                yield batch

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=tf.TensorSpec(
            shape=(None, *model_arch_params["image_shape"]),
            dtype=tf.float32,
        ),
    ).prefetch(tf.data.AUTOTUNE)


def _load_existing_weights(flow_model, run_params, model_arch_params):
    """Load weights from a completed training run for inference."""
    print(
        f"Loading model weights from file in {run_params['model_dir']}.\n",
        flush=True,
    )
    weights_path = os.path.join(
        run_params["model_dir"], "model_weights.weights.h5"
    )
    # Glow creates variables lazily on the first forward pass; they must
    # exist before load_weights can populate them.
    _ = flow_model(tf.random.normal([1, *model_arch_params["image_shape"]]))
    flow_model.load_weights(weights_path)
    _capture_and_save_summary(
        flow_model,
        model_arch_params["image_shape"],
        run_params["output_dir"],
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def default_training_sequence(  # noqa: C901
    train_gen, run_params, training_params, model_arch_params
):
    """Orchestrate model construction, optional training, and weight saving.

    Accepts the same arguments as the original flow_model.default_training_sequence
    and returns (flow_model, history) where history is None for inference runs.

    When tracking_tool='mlflow', the MLflow run is left open on return so the
    calling script can log domain-specific post-training metrics and artifacts
    before calling mlflow.end_run() itself.  run_params['mlflow_run_open'] is
    set to True to signal this to the caller.
    """
    platform = detect_platform()
    ckpt_dir = resolve_checkpoint_dir(run_params, platform)

    flow_model = FlowModel(
        **model_arch_params,
        reg_level=training_params["reg_level"],
        grad_norm_thresh=training_params["grad_norm_thresh"],
        log_scale_clip=training_params.get("log_scale_clip"),
    )
    flow_model.build(input_shape=(None, *model_arch_params["image_shape"]))
    print("")

    tracking_active = _start_tracking(
        run_params, training_params, model_arch_params, ckpt_dir, platform
    )
    _validate_weights_dest(training_params)

    history = None
    if run_params["do_train"]:
        print("Training model:", flush=True)

        optimizer, jit_compile = _build_optimizer(training_params, model_arch_params)
        flow_model.compile(optimizer=optimizer, jit_compile=jit_compile)

        initial_epoch = 0
        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            initial_epoch = _load_checkpoint(flow_model, ckpt_dir)

        callbacks = _build_callbacks(
            training_params, training_params.get("tracking_tool"), ckpt_dir
        )
        train_dataset = _make_dataset(train_gen, model_arch_params)
        steps_per_epoch = (
            training_params["num_data_input"]
            // training_params["batch_size"]
            * training_params["augmentation_factor"]
        )
        history = flow_model.fit(
            x=train_dataset,
            epochs=training_params["num_epochs"],
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=2 if platform == "sagemaker" else 1,
        )
        print("Done training model.", flush=True)

        _save_training_outputs(
            flow_model,
            run_params,
            training_params,
            model_arch_params,
            ckpt_dir,
            tracking_active,
        )
    else:
        _load_existing_weights(flow_model, run_params, model_arch_params)
        if tracking_active:
            run_params["mlflow_run_open"] = True

    return flow_model, history
