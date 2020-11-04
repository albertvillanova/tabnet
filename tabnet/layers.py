import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit activation function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        half = inputs.shape[1] // 2
        return inputs[:, :half] * tf.keras.activations.sigmoid(inputs[:, half:])


class Transform(tf.keras.layers.Layer):
    """Transform block."""
    def __init__(self, feature_dim, batch_momentum, virtual_batch_size, fc=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size

        self.fc = fc
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum, virtual_batch_size=self.virtual_batch_size)
        self.glu = GLU()

    def build(self, input_shape):
        if not self.fc:
            self.fc = tf.keras.layers.Dense(self.feature_dim * 2, use_bias=False, input_shape=input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc(x)
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
