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

Two training modes, selected by the y_dim constructor argument:

  y_dim == 0 (default; train_flowmodels1.py, train_flowmodels2.py):
      plain density estimation, x <-> z, trained by maximum likelihood.

  y_dim > 0 (train_flowmodels3.py - train_flowmodels7.py):
      "INN" mode, x <-> [y, z], where the first y_dim components on the latent
      side of the flow are trained to predict known targets y (labels, or
      data of a forward problem) and the remaining components z are Gaussian
      latents.  Trained on (x, y) pairs with a weighted sum of the losses of
      Ardizzone et al 2019 (supervised y loss + MMD losses) and a likelihood
      term in the spirit of Zhang & Curtis 2021.  See FlowModel._inn_losses.
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
        y_dim=0,
        z_dim=None,
        x_dim=None,
        lambda_y=1.0,
        lambda_z=1.0,
        lambda_x=1.0,
        lambda_pad=1.0,
        lambda_nll=0.0,
        y_sigma=1.0,
        pad_noise_scale=0.05,
        mmd_bandwidths=(0.2, 1.5, 3.0),
    ):
        """RealNVP-based flow architecture, using TFP as much as possible so the
        architectures don't *exactly* match the papers but are pretty close.

        INN-mode arguments (all ignored when y_dim == 0):
          y_dim: number of leading latent-side components that represent y.
          z_dim: number of Gaussian latent components following y; default is
              all remaining components.  Any components left after y and z
              are output-side padding.
          x_dim: number of leading data-side components that hold the real
              input x; default is all of them.  Any remaining components are
              input-side padding.  (Not used with the glow bijector.)
          lambda_y: weight of supervised loss  mean((y_pred - y_true)^2).
          lambda_z: weight of MMD[q(y, z), p(y) p(z)]   (Ardizzone L_z).
          lambda_x: weight of MMD[q(x), p(x)] from a backward pass
              (Ardizzone L_x).
          lambda_pad: weight of the padding losses (only if padding exists).
          lambda_nll: weight of the likelihood term, per dimension:
              -log q(x | y_true) / D, where on the latent side
              y ~ N(y_true, y_sigma^2), z ~ N(0, I).
          y_sigma: std of the y components about their targets in the
              likelihood term (and in sample_posterior for likelihood-trained
              models).
          pad_noise_scale: amplitude of the noise written into padding
              components (padding is "nearly zero", not exactly zero).
          mmd_bandwidths: widths h of the inverse multiquadratic kernels
              k(a, b) = h^2 / (h^2 + |a - b|^2) summed in the MMD.
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

        # INN-mode bookkeeping (y_dim == 0 leaves the original behavior):
        self.flat_dim = int(flat_image_size)
        self.y_dim = int(y_dim or 0)
        if z_dim is None:
            self.z_dim = self.flat_dim - self.y_dim
        else:
            self.z_dim = int(z_dim)
        self.x_dim = self.flat_dim if x_dim is None else int(x_dim)
        self.out_pad_dim = self.flat_dim - self.y_dim - self.z_dim
        self.x_pad_dim = self.flat_dim - self.x_dim
        self.lambda_y = float(lambda_y)
        self.lambda_z = float(lambda_z)
        self.lambda_x = float(lambda_x)
        self.lambda_pad = float(lambda_pad)
        self.lambda_nll = float(lambda_nll)
        self.y_sigma = float(y_sigma)
        self.pad_noise_scale = float(pad_noise_scale)
        self.mmd_bandwidths = tuple(float(h) for h in mmd_bandwidths)
        if self.y_dim > 0:
            if self.z_dim < 1 or self.out_pad_dim < 0:
                raise ValueError(
                    "Need y_dim + z_dim <= prod(image_shape) and z_dim >= 1; "
                    f"got y_dim={self.y_dim}, z_dim={self.z_dim}, "
                    f"prod(image_shape)={self.flat_dim}."
                )
            if not 1 <= self.x_dim <= self.flat_dim:
                raise ValueError(
                    f"Need 1 <= x_dim <= prod(image_shape); got {self.x_dim}."
                )
            if bijector == "glow" and self.x_pad_dim > 0:
                raise ValueError("x_dim padding is not supported with glow.")

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
                "y_dim": int(self.y_dim),
                "z_dim": int(self.z_dim),
                "x_dim": int(self.x_dim),
                "lambda_y": self.lambda_y,
                "lambda_z": self.lambda_z,
                "lambda_x": self.lambda_x,
                "lambda_pad": self.lambda_pad,
                "lambda_nll": self.lambda_nll,
                "y_sigma": self.y_sigma,
                "pad_noise_scale": self.pad_noise_scale,
                "mmd_bandwidths": [float(h) for h in self.mmd_bandwidths],
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

        In INN mode (y_dim > 0) data must be an (x, y) pair and the loss is
        the weighted sum described in _inn_losses.
        """
        if self.y_dim > 0:
            return self._inn_train_step(data)
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

    # ------------------------------------------------------------------
    # INN mode:  x <-> [y, z]   (used when y_dim > 0)
    # ------------------------------------------------------------------
    # Naming: "data side" is where x lives (v = [x, input padding]) and
    # "latent side" is where u = [y, z, output padding] lives.  In TFP terms
    # data -> latent is flow_bijector.inverse and latent -> data is
    # flow_bijector.forward, exactly as in call() and inverse() above.

    def _data_to_latent(self, v):
        """Map data-side batch v to a flat latent-side batch u.  Returns
        (v reshaped as the bijector expects it, u)."""
        if self.bijector_type == "glow":
            v = tf.reshape(v, (-1, *self.image_shape))
            result = self.flow_bijector.inverse(v)
            batch_size = tf.shape(v)[0]
            u = tf.concat(
                [tf.reshape(t, (batch_size, -1)) for t in tf.nest.flatten(result)],
                axis=-1,
            )
        else:
            v = tf.reshape(v, (-1, self.flat_dim))
            u = self.flow_bijector.inverse(v)
        return v, u

    def split_latent(self, u):
        """Split flat latent-side batch into (y, z, output padding)."""
        y = u[:, : self.y_dim]
        z = u[:, self.y_dim : self.y_dim + self.z_dim]
        pad = u[:, self.y_dim + self.z_dim :]
        return y, z, pad

    def _mmd(self, a, b):
        """Maximum mean discrepancy between sample sets a and b (rows are
        samples), with a sum of inverse multiquadratic kernels."""
        def sq_dists(p, q):
            pp = tf.reduce_sum(tf.square(p), axis=1, keepdims=True)
            qq = tf.reduce_sum(tf.square(q), axis=1, keepdims=True)
            d = pp + tf.transpose(qq) - 2.0 * tf.matmul(p, q, transpose_b=True)
            return tf.maximum(d, 0.0)

        daa = sq_dists(a, a)
        dbb = sq_dists(b, b)
        dab = sq_dists(a, b)
        mmd = 0.0
        for h in self.mmd_bandwidths:
            h2 = h * h
            mmd += (
                tf.reduce_mean(h2 / (h2 + daa))
                + tf.reduce_mean(h2 / (h2 + dbb))
                - 2.0 * tf.reduce_mean(h2 / (h2 + dab))
            )
        return mmd

    def _inn_losses(self, v, y_true):
        """Losses for one batch of (data-side inputs v, targets y_true).

        total = lambda_y   * mean (y_pred - y_true)^2              [L_y]
              + lambda_z   * MMD[(stopgrad(y_pred), z), (y_true, z')]  [L_z]
              + lambda_x   * MMD[x_backward, x]                     [L_x]
              + lambda_pad * padding losses
              + lambda_nll * NLL / D
        where z' ~ N(0, I), x_backward = f^-1(y_true, z'), and
          NLL = -log q(v | y_true)
              = -log N(y_pred; y_true, y_sigma^2 I) - log N(z; 0, I)
                - log N(pad; 0, pad_noise_scale^2 I) - log|det d u / d v|.
        L_y, L_z, L_x and the padding losses follow Ardizzone et al 2019
        Sec 3.3; the NLL term is the exact likelihood of a flow whose latent
        y-components are centered on the targets, which (like the likelihood
        term of Zhang & Curtis 2021 eqn 13) keeps training effective in high
        dimensions where MMD is not.

        Returns (total, dict of the individual terms).
        """
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, self.y_dim))
        v = tf.reshape(tf.cast(v, tf.float32), (-1, self.flat_dim))
        batch = tf.shape(v)[0]
        x_true = v[:, : self.x_dim]
        if self.x_pad_dim > 0:
            # overwrite input padding with low-amplitude noise
            v = tf.concat(
                [
                    x_true,
                    self.pad_noise_scale
                    * tf.random.normal((batch, self.x_pad_dim)),
                ],
                axis=-1,
            )

        # forward pass: data side -> latent side
        v_shaped, u = self._data_to_latent(v)
        y_pred, z_pred, pad_pred = self.split_latent(u)

        loss_y = tf.reduce_mean(tf.square(y_pred - y_true))

        # likelihood term (always computed, as it is also a useful metric)
        event_ndims = 3 if self.bijector_type == "glow" else 1
        ildj = self.flow_bijector.inverse_log_det_jacobian(
            v_shaped, event_ndims=event_ndims
        )
        ildj = tf.reshape(ildj, (-1,))
        nll = (
            0.5 * tf.reduce_sum(
                tf.square((y_pred - y_true) / self.y_sigma), axis=1
            )
            + self.y_dim * float(np.log(self.y_sigma))
            + 0.5 * tf.reduce_sum(tf.square(z_pred), axis=1)
            + 0.5 * self.flat_dim * float(np.log(2.0 * np.pi))
            - ildj
        )
        if self.out_pad_dim > 0:
            nll += (
                0.5 * tf.reduce_sum(
                    tf.square(pad_pred / self.pad_noise_scale), axis=1
                )
                + self.out_pad_dim * float(np.log(self.pad_noise_scale))
            )
        nll_per_dim = tf.reduce_mean(nll) / self.flat_dim

        zero = tf.constant(0.0)
        loss_z = zero
        loss_x = zero
        loss_pad = zero
        has_pad = self.out_pad_dim > 0 or self.x_pad_dim > 0
        z_samp = tf.random.normal((batch, self.z_dim))

        if self.lambda_z > 0:
            loss_z = self._mmd(
                tf.concat([tf.stop_gradient(y_pred), z_pred], axis=-1),
                tf.concat([y_true, z_samp], axis=-1),
            )

        if self.out_pad_dim > 0:
            loss_pad += tf.reduce_mean(tf.square(pad_pred))

        if self.lambda_x > 0 or (self.lambda_pad > 0 and has_pad):
            # backward pass: latent side -> data side, from true y and
            # freshly sampled z
            parts = [y_true, z_samp]
            if self.out_pad_dim > 0:
                parts.append(
                    self.pad_noise_scale
                    * tf.random.normal((batch, self.out_pad_dim))
                )
            v_back = tf.reshape(
                self.flow_bijector.forward(tf.concat(parts, axis=-1)),
                (-1, self.flat_dim),
            )
            if self.lambda_x > 0:
                loss_x = self._mmd(v_back[:, : self.x_dim], x_true)
            if self.x_pad_dim > 0:
                loss_pad += tf.reduce_mean(tf.square(v_back[:, self.x_dim :]))
            if self.out_pad_dim > 0 and self.lambda_pad > 0:
                # reconstruction with the output padding overwritten by
                # noise, so that no information gets stored in the padding
                u_noisy = tf.concat(
                    [
                        y_pred,
                        z_pred,
                        self.pad_noise_scale
                        * tf.random.normal((batch, self.out_pad_dim)),
                    ],
                    axis=-1,
                )
                v_rec = tf.reshape(
                    self.flow_bijector.forward(u_noisy), (-1, self.flat_dim)
                )
                loss_pad += tf.reduce_mean(
                    tf.square(v_rec[:, : self.x_dim] - x_true)
                )

        total = (
            self.lambda_y * loss_y
            + self.lambda_z * loss_z
            + self.lambda_x * loss_x
            + self.lambda_pad * loss_pad
            + self.lambda_nll * nll_per_dim
        )
        terms = {
            "loss_y": loss_y,
            "loss_z": loss_z,
            "loss_x": loss_x,
            "loss_pad": loss_pad,
            "nll_per_dim": nll_per_dim,
        }
        return total, terms

    def _inn_train_step(self, data):
        """Training step for INN mode; data is an (x, y) pair of batches."""
        if not isinstance(data, (tuple, list)) or len(data) < 2:
            raise ValueError(
                "FlowModel with y_dim > 0 must be trained on (x, y) pairs."
            )
        with tf.GradientTape() as tape:
            total, terms = self._inn_losses(data[0], data[1])
        tf.debugging.assert_all_finite(total, "NaN or Inf detected in INN loss")

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total, trainable_vars)
        for grad in gradients:
            if grad is None:
                continue
            tf.debugging.assert_all_finite(
                grad, "NaN or Inf detected in gradients"
            )
        present = [g for g in gradients if g is not None]
        preclip_grad_norm = tf.linalg.global_norm(present)
        if self.grad_norm_thresh is not None:
            present, _ = tf.clip_by_global_norm(
                present, clip_norm=self.grad_norm_thresh
            )
        postclip_grad_norm = tf.linalg.global_norm(present)
        present_vars = [
            v for g, v in zip(gradients, trainable_vars) if g is not None
        ]
        if present:
            self.optimizer.apply_gradients(zip(present, present_vars))

        outdict = {
            "loss": total,
            # same definition as in plain mode, from the likelihood term:
            "bits_per_dim": terms["nll_per_dim"] / tf.math.log(2.0),
        }
        outdict.update(terms)
        if self.grad_norm_thresh is not None:
            outdict.update({
                "preclip_grad_norm": preclip_grad_norm,
                "postclip_grad_norm": postclip_grad_norm,
            })
        else:
            outdict.update({"grad_norm": postclip_grad_norm})
        current_lr = self.optimizer.learning_rate
        if callable(current_lr):
            current_lr = current_lr(self.optimizer.iterations)
        outdict["learning_rate"] = tf.convert_to_tensor(current_lr)
        return outdict

    def pad_inputs(self, x):
        """Zero-pad a numpy batch of x vectors (n, x_dim) to (n, flat_dim)."""
        x = np.asarray(x, dtype=np.float32)
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] == self.flat_dim:
            return x
        out = np.zeros((x.shape[0], self.flat_dim), dtype=np.float32)
        out[:, : x.shape[1]] = x
        return out

    def forward_yz(self, x, batch_size=1000):
        """Map a numpy batch of inputs x through the model in the forward
        (data -> latent) direction.  x may be unpadded (n, x_dim), padded
        (n, flat_dim), or image-shaped.  Returns numpy arrays
        (y_pred, z, output padding)."""
        x = self.pad_inputs(x)
        chunks = []
        for i in range(0, x.shape[0], batch_size):
            chunk = tf.convert_to_tensor(x[i : i + batch_size])
            if self.bijector_type == "glow":
                chunk = tf.reshape(chunk, (-1, *self.image_shape))
            chunks.append(self.call(chunk).numpy())
        u = np.concatenate(chunks, axis=0)
        return self.split_latent(u)

    def sample_posterior(
        self,
        y_star,
        num_samples=None,
        z=None,
        temperature=1.0,
        y_noise_scale=None,
        batch_size=1000,
    ):
        """Sample x ~ q(x | y*) by running the model backward from
        [y*, z, padding] with z ~ N(0, temperature^2 I).

        y_star: one target (y_dim,) repeated num_samples times, or an array
            (n, y_dim) giving one target per sample.
        z: optional array (n, z_dim) of latents to use instead of sampling.
        y_noise_scale: std of Gaussian jitter added to y*; default is
            y_sigma for likelihood-trained models (lambda_nll > 0, where y
            has a Gaussian of that width about its target) and 0 otherwise.
        Returns a numpy array (n, x_dim), or (n, *image_shape) for glow.
        """
        y = np.atleast_2d(np.asarray(y_star, dtype=np.float32))
        if y.shape[0] == 1 and num_samples is not None:
            y = np.repeat(y, int(num_samples), axis=0)
        n = y.shape[0]
        if z is None:
            z = np.random.normal(size=(n, self.z_dim)) * temperature
        z = np.asarray(z, dtype=np.float32).reshape(n, self.z_dim)
        if y_noise_scale is None:
            y_noise_scale = self.y_sigma if self.lambda_nll > 0 else 0.0
        if y_noise_scale > 0:
            y = y + (
                np.random.normal(size=y.shape) * y_noise_scale * temperature
            ).astype(np.float32)
        parts = [y, z]
        if self.out_pad_dim > 0:
            parts.append(
                np.random.normal(size=(n, self.out_pad_dim))
                * self.pad_noise_scale
            )
        u = np.concatenate(parts, axis=1).astype(np.float32)
        chunks = []
        for i in range(0, n, batch_size):
            chunks.append(
                self.inverse(tf.convert_to_tensor(u[i : i + batch_size])).numpy()
            )
        v = np.concatenate(chunks, axis=0)
        if self.bijector_type == "glow":
            return v.reshape(n, *self.image_shape)
        return v.reshape(n, self.flat_dim)[:, : self.x_dim]
