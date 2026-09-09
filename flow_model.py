"""
Normalizing flow model definition (RealNVP and Glow variants).

Training orchestration has moved to model_training.py.  Typical usage:

    from file_utils import get_data_generator
    from model_training import default_training_sequence, load_param_overrides

    load_param_overrides(run_params, training_params, model_arch_params)
    train_gen = get_data_generator(...)
    flow_model, history = default_training_sequence(
        train_gen, run_params, training_params, model_arch_params
    )
"""

import functools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


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
            conv_layers.append(tf.keras.layers.LeakyReLU(negative_slope=leakyrelualpha))
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
                tf.keras.layers.LeakyReLU(negative_slope=leakyrelualpha)
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
                        # validate_args=True to enable shape/value checks for debugging
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

            self.flow_bijector = tfb.Chain(
                list(reversed(flow_step_list)), name=layer_name
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

    @tf.function(reduce_retracing=True)
    def sample(self, num_samples=1):
        return self.flow.sample(num_samples)

    @tf.function(reduce_retracing=True)
    def log_prob(self, x):
        return self.flow.log_prob(x)

    @tf.function(reduce_retracing=True)
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

    @tf.function(reduce_retracing=True)
    def inverse(self, outputs):
        """Gaussian latent points to images."""
        return self.flow.bijector.forward(outputs)

    @tf.function(reduce_retracing=True)
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
