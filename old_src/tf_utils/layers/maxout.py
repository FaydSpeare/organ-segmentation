import tensorflow as tf

class Maxout(tf.keras.layers.Layer):

    def call(self, inputs, training=None, mask=None):
        x = tf.stack(inputs, axis=-1)
        x = tf.reduce_max(x, axis=-1)
        return x


