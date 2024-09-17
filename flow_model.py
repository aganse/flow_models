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
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.callbacks import TensorBoard

from file_utils import infinite_generator


class FlowModel(tf.Module):
    """
    code generally follows Tensorflow Probability documentation at:
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP

    # usage with example params
    image_shape = (256, 256, 3)
    hidden_layers = [256, 256]
    flow_steps = 4
    validate_args = True
    flow_model = FlowModel(image_shape, hidden_layers, flow_steps, validate_args)
    """

    def __init__(
        self,
        image_shape=(256, 256, 3),
        hidden_layers=[256, 256],
        flow_steps=4,
        validate_args=False,
        reg_level=0.01,
    ):
        """RealNVP-based flow architecture, using TFP as much as possible so the
        architectures don't *exactly* match the papers but are pretty close.
        Refs:
        --------
        RealNVP paper:  https://arxiv.org/pdf/1605.08803
        A RealNVP tutorial found in Github:  https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/tips/RealNVP_mnist_en.ipynb
        NICE paper:  https://arxiv.org/pdf/1410.8516
        Kang ISSP 2020 paper on NICE INNs:  https://jaekookang.me/issp2020/
        Glow paper:  https://arxiv.org/pdf/1807.03039
        Eric Jang Normalizing Flows Tutorial:  https://blog.evjang.com/2018/01/nf2.html
        tfp.bijectors.RealNVP api documentation:  https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP
        Ardizzone 2019 INNs paper:  https://arxiv.org/pdf/1808.04730
        Lilian Weng Flow-based Deep Generative Models tutorial:  http://lilianweng.github.io/posts/2018-10-13-flow-models
        Jaekoo Kang's flow_based_models NICE & RealNVP repo:  https://github.com/jaekookang/flow_based_models
        Jaekoo Kang's INNs repo (Ardizzone implementation):  https://github.com/jaekookang/invertible_neural_networks
        Chanseok Kang's RealNVP notebook:
          https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2021-09-08-01-AutoRegressive-flows-and-RealNVP.ipynb#scrollTo=NNun_3RT3A56
        RealNVP implementation example in Stackoverflow:
          https://stackoverflow.com/questions/57261612/better-way-of-building-realnvp-layer-in-tensorflow-2-0
        Brian Keng's Normalizing Flows with Real NVP article, more mathematical:
          https://bjlkeng.io/posts/normalizing-flows-with-real-nvp/#modified-batch-normalization
        Helpful rundown of bits-per-dimension in Papamakarios et al 2018 paper
          "Masked Autoregressive Flow for Density Estimation": https://arxiv.org/pdf/1705.07057
          section E.2; note they call it "bits per pixel".  They express in
          average log likelihoods too (note that's actually what the NLL value
          is at very bottom of this script here).

        Note in NICE paper regarding flow_steps: "Examining the Jacobian, we
        observe that at least three coupling layers are necessary to allow all
        dimensions to influence one another. We generally use four."  And they
        used 1000-5000 nodes in their hidden layers, with 4-5 hidden layers per
        coupling layer.
        """

        super().__init__()
        self.image_shape = image_shape
        self.optimizer = None
        flat_image_size = np.prod(image_shape)  # flattened size

        layer_name = "flow_step"
        flow_step_list = []
        for i in range(flow_steps):
            # flow_step_list.append(
            #     tfp.bijectors.BatchNormalization(
            #         validate_args=validate_args,
            #         name="{}_{}_batchnorm".format(layer_name, i),
            #     )
            # )
            flow_step_list.append(
                tfp.bijectors.Permute(
                    permutation=list(reversed(range(flat_image_size))),
                    # permutation=list(np.random.permutation(flat_image_size)),
                    validate_args=validate_args,
                    name="{}_{}_permute".format(layer_name, i),
                )
            )
            flow_step_list.append(
                tfp.bijectors.RealNVP(
                    num_masked=flat_image_size // 2,
                    # shift_and_log_scale_fn=lambda x, output_dim: shift_and_log_scale_fn(x, output_dim),
                    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                        hidden_layers=hidden_layers,
                        # kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        # kernel_regularizer=tf.keras.regularizers.l2(reg_level),
                    ),
                    validate_args=validate_args,
                    name="{}_{}_realnvp".format(layer_name, i),
                )
            )
        flow_step_list = list(flow_step_list[1:])  # leave off last permute
        self.flow_bijector = tfp.bijectors.Chain(
            flow_step_list, validate_args=validate_args, name=layer_name
        )
        print(
            "flow_step_list (from input to output):",
            list(reversed([layer.name for layer in flow_step_list])),
        )

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0] * flat_image_size
        )
        self.flow = tfp.distributions.TransformedDistribution(
            distribution=base_distribution,
            bijector=self.flow_bijector,
            name="Top_Level_Flow_Model",
        )

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

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
        return {"loss": neg_log_likelihood, "bits_per_dim": bpd}

    @tf.function
    def fit(self, x, epochs, steps_per_epoch, callbacks=None):
        history = {"loss": []}

        # Initialize callbacks if any
        if callbacks:
            for callback in callbacks:
                callback.on_train_begin()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0

            # Loop over steps per epoch
            for step in range(steps_per_epoch):
                # Generate data batch from generator
                data_batch = next(x)
                loss = self.train_step(data_batch)["loss"]
                epoch_loss += loss

                # Handle batch end callbacks if any
                if callbacks:
                    for callback in callbacks:
                        callback.on_batch_end(step, logs={'loss': loss})

            # Average loss over the steps in the epoch
            epoch_loss /= steps_per_epoch
            history["loss"].append(epoch_loss)

            print(f" - loss: {epoch_loss}")

            # Handle epoch end callbacks if any
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs={"loss": epoch_loss})

        # Handle train end callbacks if any
        # if callbacks:
        #     for callback in callbacks:
        #         callback.on_train_end()

        return history


def default_training_sequence(train_gen, run_params, training_params, model_arch_params):
    """A prefab training configuration for flow_models to speed/ease getting going,
    especially as I found that Keras and TFP don't play totally well together."""

    # arg, keras and tfp are not mixing well in this original version:
    flow_model = FlowModel(**model_arch_params, reg_level=training_params["reg_level"])

    # # trying separating out the tfp calls inside build_trainable_distribution:
    # x_ = tf.keras.Input(shape=(model_arch_params["image_shape"]), dtype=tf.float32)
    # print("image_shape 1:", x_.shape)
    # reshaped_x_ = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, np.prod(model_arch_params["image_shape"]))))(x_)
    # print("image_shape 2:", reshaped_x_.shape)
    # trainable_distribution = build_trainable_distribution(
    #     model_arch_params["image_shape"],
    #     model_arch_params["flow_steps"],
    #     model_arch_params["hidden_layers"],
    #     model_arch_params["validate_args"],
    # )
    # print("reshaped_x_.shape:", reshaped_x_.shape)
    # log_prob_ = trainable_distribution.log_prob(reshaped_x_)
    # training_wrapper_model = tf.keras.Model(x_, log_prob_)

    print("")
    # Model.build() is only necessary when calling .summary() before train:
    # trainable_distribution.build(input_shape=(None, *model_arch_params["image_shape"]))
    # trainable_distribution.summary()
    # print('trainable_variables: ', trainable_distribution.variables)
    # flow_model.summary()

    if run_params["do_train"]:
        print("Training model...", flush=True)

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

        # training_wrapper_model.compile(optimizer=Adam(learning_rate=lrate), loss=lambda _, log_prob: -log_prob)
        # flow_model.compile(optimizer=Adam(learning_rate=lrate), loss=lambda _, log_prob: -log_prob)
        flow_model.set_optimizer(tf.optimizers.Adam(learning_rate=lrate))

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
        # history = training_wrapper_model.fit(
        history = flow_model.fit(
            x=infinite_train_generator,
            epochs=training_params["num_epochs"],
            steps_per_epoch=training_params["num_data_input"]
            // training_params["batch_size"]
            * training_params["augmentation_factor"],
            callbacks=callbacks,
        )
        print("Done training model.", flush=True)
        # trainable_distribution.save_weights(run_params["model_dir"] + "/model_weights")
        # print("Model weights saved to file.", flush=True)
    else:
        print(
            f"Loading model weights from file in {run_params['model_dir']}.", flush=True
        )
        flow_model.load_weights(run_params["model_dir"] + "/model_weights")

    # flow_model = TrainableDistributionContainer(trainable_distribution, model_arch_params["image_shape"])
    # print_model_summary(flow_model)
    # print('-----------------------------')
    # print_model_summary_nested(flow_model)

    return flow_model, history
