import numpy as np
import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit activation function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        half = inputs.shape[1] // 2
        return inputs[:, :half] * tf.keras.activations.sigmoid(
            inputs[:, half:])


class Transform(tf.keras.layers.Layer):
    """Transform block."""
    def __init__(self, feature_dim, batch_momentum, virtual_batch_size,
                 fc=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size

        self.fc = fc
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self.batch_momentum,
            virtual_batch_size=self.virtual_batch_size)
        self.glu = GLU()

    def build(self, input_shape):
        if not self.fc:
            self.fc = tf.keras.layers.Dense(
                self.feature_dim * 2, use_bias=False, input_shape=input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc(x)
        # TODO: self.bn.virtual_batch_size = 1 if not training
        x = self.bn(x, training=training)
        x = self.glu(x)
        return x

    def get_config(self):
        config = {
            'feature_dim': self.feature_dim,
            'batch_momentum': self.batch_momentum,
            'virtual_batch_size': self.virtual_batch_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class FeatureTransformer(tf.keras.layers.layer):
    """Feature Transformer."""
    def __init__(self, feature_dim, batch_momentum, virtual_batch_size,
                 decision_step, shared_1=None, shared_2=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.decision_step = decision_step
        self.shared_1 = shared_1
        self.shared_2 = shared_2

    def build(self, input_shape):
        self.transform_1 = Transform(self.feature_dim, self.batch_momentum,
                                     self.virtual_batch_size, fc=self.shared_1,
                                     name='transform_1',
                                     input_shape=input_shape)
        self.transform_2 = Transform(self.feature_dim, self.batch_momentum,
                                     self.virtual_batch_size, fc=self.shared_2,
                                     name='transform_2')
        self.transform_3 = Transform(self.feature_dim, self.batch_momentum,
                                     self.virtual_batch_size,
                                     name=f'transform_3_{self.decision_step}')
        self.transform_4 = Transform(self.feature_dim, self.batch_momentum,
                                     self.virtual_batch_size,
                                     name=f'transform_4_{self.decision_step}')

    def call(self, inputs, training=None):
        sqrt = np.sqrt(0.5)
        transform_1 = self.transform_1(inputs, training=training)
        transform_2 = (self.transform_2(transform_1, training=training)
                       + transform_1) * sqrt
        transform_3 = (self.transform_3(transform_2, training=training)
                       + transform_2) * sqrt
        transform_4 = (self.transform_4(transform_3, training=training)
                       + transform_3) * sqrt
        return transform_4

    def get_config(self):
        config = {
            'feature_dim': self.feature_dim,
            'batch_momentum': self.batch_momentum,
            'virtual_batch_size': self.virtual_batch_size,
            'decision_step': self.decision_step,
        }
        base_config = super().get_config()
        return {**base_config, **config}
