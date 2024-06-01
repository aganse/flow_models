"""
Usage:
flow_model = FlowModel(image_shape)
flow_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=[NegLogLikelihood()])
flow_model.fit(train_data_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# class NegLogLikelihood(tf.keras.metrics.Metric):
#     def __init__(self, name="neg_log_likelihood_metric", **kwargs):
#         super(NegLogLikelihood, self).__init__(name=name, **kwargs)
#         self.total = self.add_weight(name="total", initializer="zeros")
#         self.count = self.add_weight(name="count", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         log_prob = y_pred.log_prob(y_true)
#         # log_prob = tf.math.log(y_pred.prob(y_true) + 1e-8)  # necessary?
#         neg_log_likelihood = -tf.reduce_mean(log_prob)
#         self.total.assign_add(neg_log_likelihood)
#         self.count.assign_add(1)
#         tf.print("Log Prob:", log_prob, "NLL:", neg_log_likelihood)

#     def result(self):
#         return self.total / self.count

#     def reset_states(self):
#         self.total.assign(0)
#         self.count.assign(0)


class FlowModel(tf.keras.Model):
    """
    code generally follows Tensorflow documentation at:
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RealNVP

    # usage with example params
    image_shape = (256, 256, 3)
    hidden_layers = [256, 256]
    flow_steps = 4
    validate_args = True
    flow_model = FlowModel(image_shape, hidden_layers, flow_steps, validate_args)
    """

    def __init__(self, image_shape, hidden_layers=[256, 256], flow_steps=4, validate_args=False):
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
        Kang flow_based_models NICE & RealNVP repo:  https://github.com/jaekookang/flow_based_models
        Kang INNs repo (Ardizzone implementation):  https://github.com/jaekookang/invertible_neural_networks

        Note in NICE paper regarding flow_steps: "Examining the Jacobian, we
        observe that at least three coupling layers are necessary to allow all
        dimensions to influence one another. We generally use four."  And they
        used 1000-5000 nodes in their hidden layers, with 4-5 hidden layers per
        coupling layer.
        """

        super().__init__()
        self.image_shape = image_shape
        flat_image_size = np.prod(image_shape)  # flattened size

        layer_name = "flow_step"
        flow_step_list = []
        for i in range(flow_steps):
            # flow_step_list.append(
            #     tfp.bijectors.BatchNormalization(
            #         validate_args=validate_args,
            #         name="{}_{}/batchnorm".format(layer_name, i),
            #     )
            # )
            # flow_step_list.append(
            #     tfp.bijectors.Permute(
            #         permutation=list(range(flat_image_size)),
            #         validate_args=validate_args,
            #         name="{}_{}/permute".format(layer_name, i),
            #     )
            # )
            flow_step_list.append(
                tfp.bijectors.RealNVP(
                    num_masked=flat_image_size // 2,
                    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(hidden_layers=hidden_layers),
                    validate_args=validate_args,
                    name="{}_{}/realnvp".format(layer_name, i),
                )
            )
        self.flow_bijector = tfp.bijectors.Chain(
            list(reversed(flow_step_list)),
            validate_args=validate_args,
            name=layer_name
        )

        base_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.] * flat_image_size)
        self.flow = tfp.distributions.TransformedDistribution(
            distribution=base_distribution,
            bijector=self.flow_bijector,
            name="Top_Level_Flow_Model"
        )

    @tf.function
    def call(self, inputs):
        """Images to gaussian points"""
        inputs = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
        return self.flow.bijector.forward(inputs)

    @tf.function
    def inverse(self, outputs):
        """Gaussian points to images.
        """
        return self.flow.bijector.inverse(outputs)

    @tf.function
    def train_step(self, data):
        images = data
        images = tf.reshape(images, (-1, np.prod(self.image_shape)))
        with tf.GradientTape() as tape:
            log_prob = self.flow.log_prob(images)
            neg_log_likelihood = -tf.reduce_mean(log_prob)
            gradients = tape.gradient(neg_log_likelihood, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        return {"neg_log_likelihood": neg_log_likelihood}

    # @tf.function
    # def train_step(self, data):
    #     debug = True

    #     images = data
    #     images = tf.reshape(images, (-1, np.prod(self.image_shape)))

    #     with tf.GradientTape() as tape:
    #         log_prob = self.flow.log_prob(images)

    #         if debug:
    #             if tf.reduce_any(tf.math.is_nan(log_prob)) or tf.reduce_any(tf.math.is_inf(log_prob)):
    #                 tf.print("NaN or Inf detected in log_prob")

    #         neg_log_likelihood = -tf.reduce_mean(log_prob)
    #         gradients = tape.gradient(neg_log_likelihood, self.flow.trainable_variables)

    #         if debug:
    #             if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) for g in gradients]):
    #                 tf.print("NaN or Inf detected in gradients")

    #         # Gradient clipping
    #         if debug:
    #             gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]

    #     self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))

    #     # if debug:
    #     #     tf.print("Neg Log Likelihood:", neg_log_likelihood)

    #     return {"neg_log_likelihood": neg_log_likelihood}