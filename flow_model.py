"""
Typical usage:
from file_utils import get_data_generator
from flow_model import default_training_sequence
train_generator = get_data_generator(dataset=run_params["dataset"], batch_size=training_params["batch_size"])
flow_model = default_training_sequence(train_generator, run_params, training_params, model_arch_params)

# (inside default_training_sequence() is):
flow_model = FlowModel(**model_arch_params, reg_level=training_params["reg_level"])
flow_model.compile(optimizer=Adam(learning_rate=0.0001), metrics=[NegLogLikelihood()])
flow_model.fit(train_data_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
"""

import functools
import glob
import io
import os
import re
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import TensorBoard
import mlflow

from file_utils import infinite_generator

tfb = tfp.bijectors
tfd = tfp.distributions


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


def _capture_and_save_summary(model, image_shape, output_dir, log_to_mlflow=False):
    """Generate model summary text, save to file, and optionally log to MLflow."""
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
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    print(summary_text, end="")

    # if log_to_mlflow and mlflow.active_run():
    #     mlflow.log_artifact(summary_path, artifact_path="reports")

    return summary_path


def _patch_tfp_prefer_static_concat():
    """
    This function is a monkey-patch, included here at the top of flow_models.py
    to apply compatibility patches before anything in this module gets
    instantiated or used.  Note this function is called below, immediately after
    this function definition.

    TFP 0.25 has a bug in its internal prefer_static.concat function: when Glow's
    reshape operations run in TF graph mode, they call prefer_static.concat with
    a list that mixes int32 and int64 tensors. TF's underlying tf.concat requires
    all inputs to have the same dtype, so this mix causes a crash.

    This patch monkey-patches that internal TFP function before it can be triggered.
    It wraps the original prefer_static.concat with a version that detects the
    int32/int64 mix and casts everything to int32 first, then calls the original.
    The _glow_dtype_patch_applied guard ensures it only wraps once even if the
    module is imported multiple times.

    In short: it's a workaround for a TFP bug that only surfaces when using Glow
    in graph mode. If you ever upgrade TFP past 0.25 and the bug is fixed upstream,
    the patch is harmless (the guard and the try/except mean it degrades gracefully).
    """

    try:
        from tensorflow_probability.python.internal import prefer_static as ps
        if getattr(ps, "_glow_dtype_patch_applied", False):
            return
        _orig = ps.concat

        def _concat(values, axis, name="concat"):
            if isinstance(values, (list, tuple)):
                dtypes = {v.dtype for v in values if isinstance(v, tf.Tensor)}
                if tf.int32 in dtypes and tf.int64 in dtypes:
                    values = [
                        tf.cast(v, tf.int32)
                        if isinstance(v, tf.Tensor) and v.dtype == tf.int64
                        else v
                        for v in values
                    ]
            return _orig(values, axis, name=name)

        ps.concat = _concat
        ps._glow_dtype_patch_applied = True
    except Exception:
        pass


_patch_tfp_prefer_static_concat()


class MLflowLoggingCallback(tf.keras.callbacks.Callback):
    """Callback to mirror Keras training metrics into MLflow."""

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


class ShiftAndLogScaleCNN(tf.keras.layers.Layer):
    """A home-grown shift_and_log_scale_fn callable that's comparable to
    tfb.real_nvp_default_template but this way allows experimentation and
    expansion.
    """
    def __init__(
        self,
        output_dim,
        name=None,
        hidden_layers=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        leakyrelualpha=0.01
    ):
        super().__init__(name=name)
        self.output_dim = int(output_dim)
        conv_layers = []
        layers = hidden_layers or []
        for filters in layers:
            conv_layers.append(tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same"))
            conv_layers.append(tf.keras.layers.BatchNormalization())
            conv_layers.append(tf.keras.layers.LeakyReLU(alpha=leakyrelualpha))
        conv_layers.append(tf.keras.layers.Dense(int(2 * self.output_dim), activation=None))
        self.nn = tf.keras.Sequential(conv_layers)

    def call(self, inputs, output_units=None, **kwargs):
        del output_units
        return self.nn(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 2 * self.output_dim))


class ShiftAndLogScaleDense(tf.keras.layers.Layer):
    """A home-grown shift_and_log_scale_fn callable that's comparable to
    tfb.real_nvp_default_template but this way allows experimentation and
    expansion.
    """
    def __init__(
        self,
        output_dim,
        name=None,
        hidden_layers=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        leakyrelualpha=0.01,
        log_scale_clip=None,
    ):
        super().__init__(name=name)
        self.output_dim = int(output_dim)
        if log_scale_clip is None or log_scale_clip <= 0:
            self.log_scale_clip = None
        else:
            self.log_scale_clip = float(log_scale_clip)
        dense_layers = []
        hidden_layers = hidden_layers or []
        for nodes in hidden_layers:
            nodes = int(nodes)
            dense_layers.append(
                tf.keras.layers.Dense(
                    units=nodes,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )
            dense_layers.append(
                tf.keras.layers.LeakyReLU(alpha=leakyrelualpha)
            )
        dense_layers.append(tf.keras.layers.Dense(int(2 * self.output_dim), activation=None))
        self.nn = tf.keras.Sequential(dense_layers)

    def call(self, inputs, output_units=None, **kwargs):
        del output_units
        outputs = self.nn(inputs)
        if self.log_scale_clip is not None:
            shift, log_scale = tf.split(outputs, num_or_size_splits=2, axis=-1)
            log_scale = tf.clip_by_value(
                log_scale, -self.log_scale_clip, self.log_scale_clip
            )
            outputs = tf.concat([shift, log_scale], axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 2 * self.output_dim))


@tf.keras.utils.register_keras_serializable(package="flow_model")
class FlowModel(tf.keras.Model):
    """
    Variations of normalizing flow models including RealNVP and Glow;
    code generally follows Tensorflow Probability documentation at:
    https://www.tensorflow.org/probability/api_docs/python/tfb/RealNVP
    """

    def __init__(
        self,
        image_shape=(256, 256, 3),
        bijector="realnvp-based",  # "realnvp-based" or "glow"
        realnvp_flow_steps=4,
        realnvp_hidden_layers=None,
        realnvp_permutation="alternating",
        glow_num_blocks=3,
        glow_steps_per_block=8,
        glow_num_hidden=256,
        validate_args=False,
        grad_norm_thresh=None,
        reg_level=0.01,
        log_scale_clip=None,
    ):
        """RealNVP-based flow architecture, using TFP as much as possible so the
        architectures don't *exactly* match the papers but are pretty close.
        """

        super().__init__()
        self.image_shape = image_shape
        self.bijector_type = bijector
        self.validate_args = validate_args
        self.realnvp_flow_steps = realnvp_flow_steps
        self.realnvp_hidden_layers = list(realnvp_hidden_layers) if realnvp_hidden_layers is not None else []
        self.realnvp_permutation = realnvp_permutation
        self.glow_num_blocks = glow_num_blocks
        self.glow_steps_per_block = glow_steps_per_block
        self.glow_num_hidden = glow_num_hidden
        self.grad_norm_thresh = grad_norm_thresh
        self.reg_level = reg_level
        self.log_scale_clip = (
            None if log_scale_clip is None or log_scale_clip <= 0 else float(log_scale_clip)
        )
        self.shift_and_log_scale_layers = []
        flat_image_size = np.prod(image_shape)  # flattened size

        if bijector == "glow":

            self.flow_bijector = tfb.Glow(
                output_shape=tuple(int(x) for x in self.image_shape),
                num_glow_blocks=glow_num_blocks,
                num_steps_per_block=glow_steps_per_block,
                coupling_bijector_fn=functools.partial(
                    tfb.GlowDefaultNetwork, num_hidden=glow_num_hidden
                ),
                exit_bijector_fn=tfb.GlowDefaultExitNetwork,
            )

        elif bijector == "realnvp-based":

            realnvp_hidden_layers = realnvp_hidden_layers or [256, 256]
            layer_name = "Flow_step"
            flow_step_list = []
            for i in range(realnvp_flow_steps):
                # shift_log_scale_layer = ShiftAndLogScaleCNN(
                shift_log_scale_layer = ShiftAndLogScaleDense(
                    output_dim=flat_image_size // 2,
                    name="{}_{}_shift_log_scale_layer".format(layer_name, i),
                    hidden_layers=realnvp_hidden_layers,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                    log_scale_clip=log_scale_clip,
                )

                def shift_log_scale_fn_factory(layer):
                    def shift_log_scale_fn(x, output_units, **unused_kwargs):
                        outputs = layer(x, output_units=output_units)
                        shift, log_scale = tf.split(outputs, num_or_size_splits=2, axis=-1)
                        return shift, log_scale
                    return shift_log_scale_fn

                shift_log_scale_fn = shift_log_scale_fn_factory(shift_log_scale_layer)
                self.shift_and_log_scale_layers.append(shift_log_scale_layer)
                flow_step_list.append(
                    tfb.RealNVP(
                        num_masked=flat_image_size // 2,
                        # (using own shift_and_log_scale_fn to experiment/expand,
                        # but similar to tfb.real_nvp_default_template)
                        shift_and_log_scale_fn=shift_log_scale_fn,
                        # shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        #    hidden_layers=hidden_layers,
                        #    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        #    kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                        # ),
                        # fyi log_scale_clip_fn doesn't exist in this version of tfb:
                        # log_scale_clip_fn=lambda log_s: tf.clip_by_value(log_s, -5.0, 5.0),
                        validate_args=validate_args,
                        name="{}_{}_RealNVP".format(layer_name, i),
                    )
                )
                if realnvp_permutation == "random":
                    perm = list(np.random.permutation(flat_image_size))
                else:  # "alternating"
                    perm = (
                        list(reversed(range(flat_image_size)))
                        if i % 2 == 0 else list(range(flat_image_size))
                    )
                flow_step_list.append(
                    tfb.Permute(
                        permutation=perm,
                        validate_args=validate_args,
                        name="{}_{}_Permute".format(layer_name, i),
                    )
                )
                # This is mentioned in paper but I can't get it to stabilize:
                # (note if using this need to use [:-2] rather than [:-1] below)
                # flow_step_list.append(
                #     tfb.BatchNormalization(
                #         validate_args=validate_args,
                #         name="{}_{}_BatchNorm".format(layer_name, i),
                #     )
                # )
            flow_step_list = flow_step_list[:-1]  # leave off last permute

            print("Flow_step_layers:")
            print("-------------------")
            print("\n".join([layer.name for layer in flow_step_list]))
            print("")

            self.flow_bijector = tfb.Chain(
                list(reversed(flow_step_list)), validate_args=validate_args, name=layer_name
            )

        base_distribution = tfd.MultivariateNormalDiag(
            loc=[0.0] * flat_image_size
        )

        self.flow = tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=self.flow_bijector,
            name="Top_Level_Flow_Model",
        )

    def get_config(self):
        """Return serializable config so `to_json` captures runtime params."""
        base_config = super().get_config()
        base_config.update(
            {
                "image_shape": tuple(self.image_shape),
                "bijector": str(self.bijector_type),
                "validate_args": bool(self.validate_args),
                "realnvp_flow_steps": int(self.realnvp_flow_steps),
                "realnvp_hidden_layers": list(self.realnvp_hidden_layers),
                "realnvp_permutation": str(self.realnvp_permutation),
                "glow_num_blocks": int(self.glow_num_blocks),
                "glow_steps_per_block": int(self.glow_steps_per_block),
                "glow_num_hidden": int(self.glow_num_hidden),
                "grad_norm_thresh": self.grad_norm_thresh,
                "reg_level": float(self.reg_level) if self.reg_level is not None else None,
                "log_scale_clip": self.log_scale_clip,
            }
        )
        return base_config

    @classmethod
    def from_config(cls, config):
        # Pop base Keras Model config entries that FlowModel.__init__ doesn't accept.
        config = dict(config)
        config.pop("name", None)
        config.pop("trainable", None)
        config.pop("dtype", None)
        return cls(**config)

    @property
    def trainable_variables(self):
        # tf.keras.Model.trainable_variables only recurses into Keras-tracked
        # sub-objects (Layers/Models). tfb.Glow is a tf.Module but not a Keras
        # Layer, so its variables are missed. Collect from both sources.
        seen = {}
        for v in super().trainable_variables:
            seen[id(v)] = v
        for v in self.flow_bijector.trainable_variables:
            seen.setdefault(id(v), v)
        return list(seen.values())

    def print_vars(self):
        """More detailed output per model layers, mainly for debugging purposes.
        """

        # To access build model layers must put one sample thru it first:
        x = tf.random.normal([1, np.prod(self.image_shape)])
        x = tf.expand_dims(x, axis=0)  # adds a batch dimension to the sample
        _ = self.flow.log_prob(x)

        # Now we can access the layers to print out:
        print("")
        print("More-detailed object listing of bijectors in the chain (from output to input):")
        print("------------------------------------------------------------------------------")
        for bijector in [self.flow.bijector]:
            print(f"Bijector: {type(bijector).__name__}")  # the chain itself
            for layer in bijector.bijectors:  # the bijector layers in chain
                print("  ", layer)
        print("")

    @tf.function
    def sample(self, num_samples=1):
        return self.flow.sample(num_samples)

    @tf.function
    def log_prob(self, x):
        return self.flow.log_prob(x)

    @tf.function
    def call(self, inputs):
        """Images to Gaussian latent points."""
        if self.bijector_type == "glow":
            inputs = tf.reshape(inputs, (-1, *self.image_shape))
            result = self.flow.bijector.inverse(inputs)
            # Glow's multi-scale exits produce a dict of per-scale latents;
            # flatten and concat into a single vector matching the base distribution.
            batch_size = tf.shape(inputs)[0]
            return tf.concat(
                [tf.reshape(t, (batch_size, -1)) for t in tf.nest.flatten(result)],
                axis=-1,
            )
        else:
            inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
            return self.flow.bijector.inverse(inputs)

    @tf.function
    def inverse(self, outputs):
        """Gaussian latent points to images."""
        return self.flow.bijector.forward(outputs)

    @tf.function
    def train_step(self, data):
        """Compute NLL and gradients for a given training step.
        Note that NLL here is actually average NLL per image (avg over N images),
        consistent with many papers in the literature, and supporting the
        bits-per-dimension value as a "within one image" value - an average
        over the current batch.
        """
        images = data[0] if isinstance(data, (tuple, list)) else data
        if self.bijector_type == "glow":
            images = tf.reshape(images, (-1, *self.image_shape))
        else:
            images = tf.reshape(images, (-1, np.prod(self.image_shape)))
        with tf.GradientTape() as tape:

            log_prob = self.flow.log_prob(images)

            tf.debugging.assert_all_finite(
                log_prob, "NaN or Inf detected in log_prob"
            )

            neg_log_likelihood = -tf.reduce_mean(log_prob)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(neg_log_likelihood, trainable_vars)

            for grad in gradients:
                if grad is None:
                    continue
                tf.debugging.assert_all_finite(
                    grad, "NaN or Inf detected in gradients"
                )

            # Gradient clipping:
            if self.grad_norm_thresh is not None:
                preclip_grad_norm = tf.linalg.global_norm(gradients)
                preclip_grad_norm = tf.reduce_mean(preclip_grad_norm)
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.grad_norm_thresh)  # scales whole gradient
                # gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # gradient direction can change
            postclip_grad_norm = tf.linalg.global_norm(gradients)

        grads_and_vars = [
            (g, v) for g, v in zip(gradients, trainable_vars) if g is not None
        ]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)

        # Assemble and output progress values to log
        bits_per_dim_divisor = np.prod(self.image_shape) * tf.math.log(2.0)
        bpd = neg_log_likelihood / bits_per_dim_divisor
        outdict = {
            "loss": neg_log_likelihood,
            "bits_per_dim": bpd,
        }
        if self.grad_norm_thresh is not None:
            outdict.update({
                "preclip_grad_norm": preclip_grad_norm,
                "postclip_grad_norm": postclip_grad_norm,
            })
        else:
            outdict.update({
                "grad_norm": postclip_grad_norm,
            })
        # if isinstance(
        #     self.optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
        # ):
        #     current_lr = self.optimizer.learning_rate(self.optimizer.iterations)
        #     outdict.update({"learning_rate": current_lr})

        current_lr = self.optimizer.learning_rate
        if callable(current_lr):
            current_lr = current_lr(self.optimizer.iterations)
        outdict["learning_rate"] = tf.convert_to_tensor(current_lr)

        return outdict


def default_training_sequence(train_gen, run_params, training_params, model_arch_params):  # noqa: C901
    """A prefab training configuration for flow_models to speed/ease getting going,
    especially as I found that Keras and TFP don't play totally well together."""

    flow_model = FlowModel(
        **model_arch_params,
        reg_level=training_params["reg_level"],
        grad_norm_thresh=training_params["grad_norm_thresh"],
        log_scale_clip=training_params.get("log_scale_clip"),
    )
    flow_model.build(input_shape=(None, *model_arch_params["image_shape"]))
    print("")

    history = None

    tracking_tool = training_params.get("tracking_tool")
    valid_tools = {None, "tensorboard", "mlflow"}
    if tracking_tool not in valid_tools:
        raise ValueError(
            f"Unsupported tracking_tool '{tracking_tool}'. Expected one of {valid_tools - {None}} or None."
        )
    tracking_port = training_params.get("tracking_port")
    mlflow_run_started = False
    if tracking_tool == "mlflow":
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
        dataset = run_params.get("dataset", "flow_model_run")
        num_gen = run_params.get("num_gen_sims", "NA")
        run_name = f"{dataset}_{num_gen}"
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_name=run_name, log_system_metrics=True)
        mlflow.set_tag("mlflow.user", os.environ.get("HOST_USER", os.environ.get("USER", "unknown")))
        mlflow.set_tag("image_tag", os.environ.get("IMAGE_TAG", "[local]"))
        params_for_logging = {
            **run_params,
            **training_params,
            **model_arch_params,
            "tracking_tool": tracking_tool,
        }
        params_for_logging.pop("output_dir", None)
        mlflow.log_params(_flatten_params(params_for_logging))
        active_run = mlflow.active_run()
        if active_run:
            run_params["mlflow_run_id"] = active_run.info.run_id
        mlflow_run_started = True

    if training_params.get("save_model_weights"):
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

    if run_params["do_train"]:
        print("Training model:", flush=True)

        if isinstance(training_params["learning_rate"], float):
            lrate = training_params["learning_rate"]
        elif (
            isinstance(training_params["learning_rate"], list)
            and len(training_params["learning_rate"]) == 3
        ):
            lrate = ExponentialDecay(
                training_params["learning_rate"][0],
                decay_steps=training_params["learning_rate"][1],
                decay_rate=training_params["learning_rate"][2],
                staircase=True,
            )
        else:
            print("train.py: error: learning_rate not scalar or list of length 3.")
            quit()

        jit_compile = training_params["jit_compile"]
        if model_arch_params.get("bijector") == "glow" and jit_compile:
            # Glow's trainable 1x1 conv creates permutation variables on CPU;
            # XLA cannot access CPU variables from GPU kernels. Also, TFP 0.25
            # Glow has int32/int64 shape issues that only fully surface under XLA.
            print("Note: jit_compile disabled for Glow bijector (XLA/TFP incompatibility).\n",
                  flush=True)
            jit_compile = False
        flow_model.compile(
            optimizer=Adam(learning_rate=lrate),
            # jit_compile=training_params["jit_compile"],
            jit_compile=jit_compile,
        )

        initial_epoch = 0
        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            _ckpt_dir = (
                "/opt/ml/checkpoints"
                if os.path.exists("/opt/ml")
                else os.path.join(run_params["model_dir"], "checkpoints")
            )
            ckpt_files = sorted(
                glob.glob(os.path.join(_ckpt_dir, "ckpt-*.weights.h5"))
            )
            if ckpt_files:
                latest = ckpt_files[-1]
                initial_epoch = int(re.search(r"ckpt-(\d+)", latest).group(1))
                flow_model.load_weights(latest)
                print(
                    f"Resuming from checkpoint epoch {initial_epoch}: {latest}\n",
                    flush=True,
                )

        callbacks = []
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
                    f"TensorBoard logs: {log_dir} (launch via `tensorboard --logdir {log_dir} --port {tracking_port}`)"
                )
            else:
                print(f"TensorBoard logs: {log_dir}")
        elif tracking_tool == "mlflow":
            callbacks.append(
                MLflowLoggingCallback()
            )
        if training_params.get("checkpoint_every_n_epochs", 0) > 0:
            _ckpt_dir = (
                "/opt/ml/checkpoints"
                if os.path.exists("/opt/ml")
                else os.path.join(run_params["model_dir"], "checkpoints")
            )
            os.makedirs(_ckpt_dir, exist_ok=True)
            _steps_per_epoch = (
                training_params["num_data_input"]
                // training_params["batch_size"]
                * training_params["augmentation_factor"]
            )
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(_ckpt_dir, "ckpt-{epoch:04d}.weights.h5"),
                    save_weights_only=True,
                    save_freq=training_params["checkpoint_every_n_epochs"] * _steps_per_epoch,
                    verbose=1,
                )
            )

        def _train_data_gen():
            for batch in infinite_generator(train_gen):
                if isinstance(batch, (tuple, list)):
                    yield batch[0]
                else:
                    yield batch

        train_dataset = tf.data.Dataset.from_generator(
            _train_data_gen,
            output_signature=tf.TensorSpec(
                shape=(None, *model_arch_params["image_shape"]),
                dtype=tf.float32
            ),
        ).prefetch(tf.data.AUTOTUNE)
        history = flow_model.fit(
            x=train_dataset,
            epochs=training_params["num_epochs"],
            steps_per_epoch=training_params["num_data_input"]
            // training_params["batch_size"]
            * training_params["augmentation_factor"],
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=2 if os.path.exists("/opt/ml") else 1,  # per-epoch in SM logs, animated locally
        )
        print("Done training model.", flush=True)
        os.makedirs(run_params["model_dir"], exist_ok=True)

        if training_params["save_model_weights"]:
            # save the model weights:
            weights_path = os.path.join(run_params["model_dir"], "model_weights.weights.h5")
            flow_model.save_weights(weights_path)
            print("Model weights saved to file.\n", flush=True)
            weights_dest = os.environ.get("WEIGHTS_PATH")
            if weights_dest:
                if weights_dest.startswith("s3://"):
                    import boto3
                    from urllib.parse import urlparse
                    parsed = urlparse(weights_dest)
                    bucket = parsed.netloc
                    key = parsed.path.lstrip("/") + "/" + os.path.basename(weights_path)
                    boto3.client("s3").upload_file(weights_path, bucket, key)
                    print(f"Model weights uploaded to s3://{bucket}/{key}\n", flush=True)
                else:
                    import shutil
                    os.makedirs(weights_dest, exist_ok=True)
                    shutil.copy2(weights_path, weights_dest)
                    print(f"Model weights copied to {weights_dest}\n", flush=True)

        # save the txt summary description of model arch:
        summary_path = _capture_and_save_summary(
            flow_model,
            model_arch_params["image_shape"],
            run_params["model_dir"],
            log_to_mlflow=mlflow_run_started,
        )
        # lastly save the model architecture to file:
        arch_path = os.path.join(run_params["model_dir"], "model_arch.json")
        with open(arch_path, "w") as f:
            f.write(flow_model.to_json())
        if mlflow_run_started:
            run_params["model_summary_path"] = summary_path
        if mlflow_run_started:
            run_params["mlflow_run_open"] = True
    else:
        print(
            f"Loading model weights from file in {run_params['model_dir']}.\n", flush=True
        )
        weights_path = os.path.join(run_params["model_dir"], "model_weights.weights.h5")
        # Glow creates variables lazily on the first forward pass; they must
        # exist before load_weights can populate them:
        _ = flow_model(tf.random.normal([1, *model_arch_params["image_shape"]]))
        flow_model.load_weights(weights_path)
        _capture_and_save_summary(
            flow_model,
            model_arch_params["image_shape"],
            run_params["output_dir"],
            log_to_mlflow=False,
        )
        if mlflow_run_started:
            run_params["mlflow_run_open"] = True

    return flow_model, history
