"""
Usage:
flow_model = FlowModel(image_shape)
flow_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=[NegLogLikelihood()])
flow_model.fit(train_data_generator, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class BijectorLayer(tf.keras.layers.Layer):
    """Wraps a TFP or home-grown bijector class as a Keras layer in order to
    get built-in benefits like outputting layer info into model.summary().
    (Experimental - not sure using this class is going to work...)
    """
    def __init__(self, bijector, **kwargs):
        super(BijectorLayer, self).__init__(**kwargs)
        self.bijector = bijector

    def call(self, inputs, **kwargs):
        return self.bijector.forward(inputs)

    def inverse(self, inputs, **kwargs):
        return self.bijector.inverse(inputs)


class ActNorm(tfp.bijectors.Bijector):
    """An activation normalization layer to use after the coupling layers
    (from RealNVP), to try to emulate Glow model architecture.
    """
    def __init__(self, image_shape, validate_args=False, name="ActNorm", **kwargs):
        super().__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs)
        self.image_shape = image_shape
        num_channels = image_shape[-1]
        self.log_scale = tf.Variable(tf.zeros(num_channels), name="log_scale")
        self.bias = tf.Variable(tf.zeros(num_channels), name="bias")

    @tf.function
    def _forward(self, x):
        x = tf.reshape(x, [-1] + list(self.image_shape))
        y = tf.nn.bias_add(x * tf.exp(self.log_scale), self.bias)
        return tf.reshape(y, [-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2]])

    @tf.function
    def _inverse(self, y):
        y = tf.reshape(y, [-1] + list(self.image_shape))
        x = (y - self.bias) * tf.exp(-self.log_scale)
        return tf.reshape(x, [-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2]])

    @tf.function
    def _forward_log_det_jacobian(self, x):
        # return tf.reduce_sum(self.log_scale) * tf.reduce_prod(tf.shape(x)[1:3])
        return tf.reduce_sum(self.log_scale) * tf.cast(tf.reduce_prod(tf.shape(x)[1:3]), tf.float32)

    @tf.function
    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(y)


class Invertible1x1Conv(tfp.bijectors.Bijector):
    """An invertible 1x1 convolutional layer to use after the coupling layers
    (from RealNVP), to try to emulate Glow model architecture.
    """
    def __init__(self, image_shape, validate_args=False, name="Invertible1x1Conv", **kwargs):
        super().__init__(forward_min_event_ndims=1, inverse_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs)
        self.image_shape = image_shape
        num_channels = image_shape[-1]
        w_init = tf.linalg.qr(tf.random.normal([num_channels, num_channels]))[0]
        self.w = tf.Variable(w_init)

    @tf.function
    def _forward(self, x):
        x = tf.reshape(x, [-1] + list(self.image_shape))  # Reshape flattened input to image format
        w_matrix = tf.reshape(self.w, [1, 1] + self.w.shape.as_list())
        y = tf.nn.conv2d(x, w_matrix, strides=[1, 1, 1, 1], padding='SAME')
        return tf.reshape(y, [-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2]])  # Flatten back

    @tf.function
    def _inverse(self, y):
        y = tf.reshape(y, [-1] + list(self.image_shape))  # Reshape flattened input to image format
        w_inv = tf.linalg.inv(self.w)
        w_matrix_inv = tf.reshape(w_inv, [1, 1] + w_inv.shape.as_list())
        x = tf.nn.conv2d(y, w_matrix_inv, strides=[1, 1, 1, 1], padding='SAME')
        return tf.reshape(x, [-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2]])  # Flatten back

    @tf.function
    def _forward_log_det_jacobian(self, x):
        return tf.math.log(tf.abs(tf.linalg.det(self.w))) * self.image_shape[0] * self.image_shape[1]


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

    def __init__(self, image_shape, hidden_layers=[256, 256], flow_steps=4, reg_level=0.01, validate_args=False):
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

        Note in NICE paper regarding flow_steps: "Examining the Jacobian, we
        observe that at least three coupling layers are necessary to allow all
        dimensions to influence one another. We generally use four."  And they
        used 1000-5000 nodes in their hidden layers, with 4-5 hidden layers per
        coupling layer.
        """

        super().__init__()
        self.conv_layers_list = []
        self.image_shape = image_shape
        flat_image_size = np.prod(image_shape)  # flattened size

        # Defining the CNN layers used in _shift_and_log_scale_conv, which can
        # only be defined once but get called on every training iteration
        # self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        # self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        # self.conv3 = tf.keras.layers.Conv2D(flat_image_size, (3, 3), padding='same')

        layer_name = "flow_step"
        flow_step_list = []
        for i in range(flow_steps):
            flow_step_list.append(
                ActNorm(
                    image_shape=self.image_shape,
                    validate_args=validate_args,
                    name="{}_{}/actnorm".format(layer_name, i),
                )
                # tfp.bijectors.BatchNormalization(
                #     validate_args=validate_args,
                #     name="{}_{}/batchnorm".format(layer_name, i),
                # )
            )
            flow_step_list.append(
                Invertible1x1Conv(
                    image_shape=self.image_shape,
                    validate_args=validate_args,
                    name="{}_{}/invertible1x1Conv".format(layer_name, i),
                )
                # tfp.bijectors.Permute(
                #     # permutation=list(reversed(range(flat_image_size))),
                #     permutation=list(np.random.permutation(flat_image_size)),
                #     validate_args=validate_args,
                #     name="{}_{}/permute".format(layer_name, i),
                # )
            )
            flow_step_list.append(
                tfp.bijectors.RealNVP(
                    num_masked=flat_image_size // 2,
                    shift_and_log_scale_fn=self._shift_and_log_scale_conv,
                    # shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                    #     hidden_layers=hidden_layers,
                    #     kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    #     kernel_regularizer=tf.keras.regularizers.l2(reg_level)
                    # ),
                    validate_args=validate_args,
                    name="{}_{}/realnvp".format(layer_name, i),
                )
            )
        flow_step_list = list(flow_step_list[1:])  # leave off last permute
        self.flow_bijector = tfp.bijectors.Chain(
            flow_step_list,
            validate_args=validate_args,
            name=layer_name
        )
        print("flow_step_list:", flow_step_list)

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
            if tf.reduce_any(tf.math.is_nan(log_prob)) or tf.reduce_any(tf.math.is_inf(log_prob)):
                tf.print("NaN or Inf detected in log_prob")
            neg_log_likelihood = -tf.reduce_mean(log_prob)
            gradients = tape.gradient(neg_log_likelihood, self.flow.trainable_variables)
            if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) for g in gradients]):
                tf.print("NaN or Inf detected in gradients")
            gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]  # gradient clipping
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        bits_per_dim_divisor = (np.prod(self.image_shape) * tf.math.log(2.0))
        bpd = neg_log_likelihood / bits_per_dim_divisor
        return {"neg_log_likelihood": neg_log_likelihood, "bits_per_dim": bpd}

    def make_conv_layers(self, output_units):
        # This method dynamically creates new layers with their own weights
        self.conv_layers_list.append([
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2 * output_units, (3, 3), padding='same')
        ])
        return self.conv_layers_list[-1]

    def _shift_and_log_scale_conv(self, x, output_units):
        # Assuming 'x' is flattened and we know the desired image shape is self.image_shape
        print(f"_shift_and_log_scale_conv: output_units={output_units}")
        image_shape = self.image_shape
        print("Original shape:", x.shape)
        x = tf.reshape(x, [-1, *image_shape])
        print("Reshaped to image shape:", x.shape)
        layers = self.make_conv_layers(output_units)
        for layer in layers:
            x = layer(x)
        # x = layers['conv1'](x)
        # x = layers['conv2'](x)
        # x = layers['conv3'](x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        # x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        # x = tf.keras.layers.Conv2D(2 * output_units, (3, 3), padding='same')(x)
        shift, log_scale = tf.split(x, num_or_size_splits=2, axis=-1)  # axis=-1 is the channels dim
        print("Shift shape:", shift.shape, "Log scale shape:", log_scale.shape)
        # Flatten back before returning
        shift = tf.reshape(shift, [-1, image_shape[0] * image_shape[1] * image_shape[2] // 2])
        log_scale = tf.reshape(log_scale, [-1, image_shape[0] * image_shape[1] * image_shape[2] // 2])
        print("After reshape:  Shift shape:", shift.shape, "Log scale shape:", log_scale.shape)
        return shift, log_scale
