import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit activation function."""
    def __init__(self, n_units):
        super().__init__()
        self.n_units = n_units

    def call(self, inputs):
        return inputs[:, :self.n_units] * tf.keras.activations.sigmoid(inputs[:, self.n_units:])

    def get_config(self):
        config = {'n_units': self.n_units}
        base_config = super().get_config()
        return {**base_config, **config}
