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

from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.callbacks import TensorBoard

from file_utils import infinite_generator

tfb = tfp.bijectors
tfd = tfp.distributions


class ShiftAndLogScaleLayer(tf.keras.layers.Layer):
    """A home-grown shift_and_log_scale_fn callable that's comparable to
    tfb.real_nvp_default_template but this way allows experimentation and
    expansion.
    """
    def __init__(
        self,
        output_dim,
        name=None,
        hidden_layers=None,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=None
    ):
        super().__init__(name=name)
        layers = []
        for nodes in hidden_layers:
            layers.append(
                tf.keras.layers.Dense(
                    nodes,
                    activation='relu',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )
        layers.append(tf.keras.layers.Dense(2 * output_dim, activation=None))
        self.nn = tf.keras.Sequential(layers)

    def call(self, x, output_units, **kwargs):
        shift_log_scale = self.nn(x)
        shift, log_scale = tf.split(shift_log_scale, num_or_size_splits=2, axis=-1)
        return shift, log_scale


class FlowModel(tf.keras.Model):
    """
    Variations of normalizing flow models including RealNVP and Glow;
    code generally follows Tensorflow Probability documentation at:
    https://www.tensorflow.org/probability/api_docs/python/tfb/RealNVP
    """

    def __init__(
        self,
        image_shape=(256, 256, 3),
        hidden_layers=[256, 256],
        flow_steps=4,
        validate_args=False,
        bijector="realnvp-based",  # or "glow"
        reg_level=0.01,
    ):
        """RealNVP-based flow architecture, using TFP as much as possible so the
        architectures don't *exactly* match the papers but are pretty close.
        """

        super().__init__()
        self.image_shape = image_shape
        self.shift_and_log_scale_layers = []
        flat_image_size = np.prod(image_shape)  # flattened size

        if bijector == "glow":

            self.flow_bijector = tfb.Glow(
                output_shape=self.image_shape,
                coupling_bijector_fn=tfb.GlowDefaultNetwork,
                exit_bijector_fn=tfb.GlowDefaultExitNetwork
            )

        elif bijector == "realnvp-based":

            layer_name = "Flow_step"
            flow_step_list = []
            for i in range(flow_steps):
                shift_log_scale_layer = ShiftAndLogScaleLayer(
                    output_dim=flat_image_size // 2,
                    name="{}_{}_shift_log_scale_layer".format(layer_name, i),
                    hidden_layers=hidden_layers,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                )
                flow_step_list.append(
                    tfb.RealNVP(
                        num_masked=flat_image_size // 2,
                        # (using own shift_and_log_scale_fn to experiment/expand,
                        # but similar to tfb.real_nvp_default_template)
                        shift_and_log_scale_fn=shift_log_scale_layer,
                        # shift_and_log_scale_fn=tfb.real_nvp_default_template(
                        #     hidden_layers=hidden_layers,
                        #     # kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        #     # kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                        # ),
                        validate_args=validate_args,
                        name="{}_{}_RealNVP".format(layer_name, i),
                    )
                )
                flow_step_list.append(
                    tfb.Permute(
                        # Simply alternating halves:
                        # permutation=(
                        #     list(reversed(range(flat_image_size)))
                        #     if i % 2 == 0 else range(flat_image_size)
                        # ),
                        permutation=list(np.random.permutation(flat_image_size)),
                        validate_args=validate_args,
                        name="{}_{}_Permute".format(layer_name, i),
                    )
                )
                # this is in paper but I can't get it to stabilize:
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
        """Images to gaussian points"""
        inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
        return self.flow.bijector.forward(inputs)

    @tf.function
    def inverse(self, outputs):
        """Gaussian points to images."""
        return self.flow.bijector.inverse(outputs)

    @tf.function
    def train_step(self, data):
        """Compute NLL and gradients for a given training step.
        Note that NLL here is actually average NLL per image (avg over N images),
        consistent with many papers in the literature, and supporting the
        bits-per-dimension value as a "within one image" value - an average
        over the current batch.
        """
        images = data
        images = tf.reshape(images, (-1, np.prod(self.image_shape)))
        with tf.GradientTape() as tape:
            log_prob = self.flow.log_prob(images)
            if tf.reduce_any(tf.math.is_nan(log_prob)) or tf.reduce_any(
                tf.math.is_inf(log_prob)
            ):
                tf.print("NaN or Inf detected in log_prob")
            neg_log_likelihood = -tf.reduce_mean(log_prob)
            gradients = tape.gradient(neg_log_likelihood, self.flow.trainable_variables)
            if tf.reduce_any(
                [
                    tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g))
                    for g in gradients
                ]
            ):
                tf.print("NaN or Inf detected in gradients")
            gradients = [
                tf.clip_by_value(g, -1.0, 1.0) for g in gradients
            ]  # gradient clipping
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        bits_per_dim_divisor = np.prod(self.image_shape) * tf.math.log(2.0)
        bpd = neg_log_likelihood / bits_per_dim_divisor
        if isinstance(self.optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = self.optimizer.lr(self.optimizer.iterations)
        else:
            current_lr = self.optimizer.lr
        return {"loss": neg_log_likelihood, "bits_per_dim": bpd, "learning_rate": current_lr}


def default_training_sequence(train_gen, run_params, training_params, model_arch_params):
    """A prefab training configuration for flow_models to speed/ease getting going,
    especially as I found that Keras and TFP don't play totally well together."""

    flow_model = FlowModel(**model_arch_params, reg_level=training_params["reg_level"])
    flow_model.build(input_shape=(None, *model_arch_params["image_shape"]))
    flow_model.summary()
    # flow_model.print_vars()
    print("")

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

        flow_model.compile(optimizer=Adam(learning_rate=lrate))

        callbacks = []
        if training_params["early_stopping_patience"] > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="neg_log_likelihood",
                    patience=training_params["early_stopping_patience"],
                    restore_best_weights=True,
                )
            )
        if run_params["use_tensorboard"]:
            log_dir = f"./logs/train/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            callbacks.append(
                TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False)
            )
        infinite_train_generator = infinite_generator(train_gen)
        history = flow_model.fit(
            x=infinite_train_generator,
            epochs=training_params["num_epochs"],
            steps_per_epoch=training_params["num_data_input"]
            // training_params["batch_size"]
            * training_params["augmentation_factor"],
            callbacks=callbacks,
        )
        print("Done training model.", flush=True)
        flow_model.save_weights(run_params["model_dir"] + "/model_weights")
        print("Model weights saved to file.\n", flush=True)
    else:
        print(
            f"Loading model weights from file in {run_params['model_dir']}.\n", flush=True
        )
        flow_model.load_weights(run_params["model_dir"] + "/model_weights")

    return flow_model, history
