import tensorflow as tf

from tf_utils.blocks import cnn_block


class DilatedConv(tf.keras.Model):

    def __init__(self, n_layers, filters, k_size, dilation_rate, dropout, kernel_initializer="glorot_uniform", batch_norm=False, activation=tf.keras.layers.LeakyReLU):
        super(DilatedConv, self).__init__(name='')

        self.n_layers = n_layers
        self.parts = []
        for i in range(n_layers):
            self.parts.append(
                cnn_block.CNN(filters, 1, kernel_initializer=kernel_initializer, dilation_rate=dilation_rate, batch_norm=batch_norm, activation=activation))
            self.parts.append(cnn_block.CNN(filters // 4, k_size, kernel_initializer=kernel_initializer,
                                            dilation_rate=dilation_rate, batch_norm=batch_norm, activation=activation))
            self.parts.append(tf.keras.layers.Dropout(dropout))
            self.parts.append(tf.keras.layers.Concatenate())

    def call(self, input_tensor, training):

        concat = input_tensor
        x = input_tensor
        for layer in self.parts:
            if "conc" in layer.name:
                x = layer([concat, x], training=training)
                concat = x
            else:
                x = layer(x, training=training)

        return x

    def summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='DilatedConv.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == "__main__":
    block = DilatedConv(2, 32, 3, 2, 0.5)
    block.summary((8, 8, 8, 1))
