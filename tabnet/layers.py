import tensorflow as tf


class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit activation function."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        half = inputs.shape[1] // 2
        return inputs[:, :half] * tf.keras.activations.sigmoid(inputs[:, half:])

