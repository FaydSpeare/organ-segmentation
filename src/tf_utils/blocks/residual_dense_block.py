import tensorflow as tf

from tf_utils.blocks import cnn_block



class ResDense(tf.keras.Model):

    def __init__(self, n_layers, filters, k_size, kernel_initializer="glorot_uniform", batch_norm=False, activation=tf.keras.layers.LeakyReLU):
        super(ResDense, self).__init__(name='')

        # Convolve input
        self.conv_input = cnn_block.CNN(filters, k_size, kernel_initializer=kernel_initializer, batch_norm=batch_norm, activation=None)

        # Dense branch
        self.n_layers = n_layers
        self.parts = []
        for i in range(n_layers):
            self.parts.append(
                cnn_block.CNN(filters, k_size, kernel_initializer=kernel_initializer, batch_norm=batch_norm, activation=activation))
            self.parts.append(tf.keras.layers.Concatenate())

        # Final section
        self.conv_resnet = cnn_block.CNN(filters, k_size, kernel_initializer=kernel_initializer, batch_norm=batch_norm, activation=None)
        self.residual = tf.keras.layers.Add()
        self.final_activation = activation()

    def call(self, input_tensor, training):

        # Convolve input
        conv_input = self.conv_input(input_tensor)

        # Dense branch
        x = input_tensor
        concats = [x]
        for i in range(self.n_layers):
            x = self.parts[i * 2](x, training=training)
            concats.append(x)
            x = self.parts[i * 2 + 1](concats)

        # Final section
        conv_resnet = self.conv_resnet(x, training=training)
        residual = self.residual([conv_input, conv_resnet])

        return self.final_activation(residual)

    def summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='ResDense.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == "__main__":
    block = ResDense(2, 32, 3, "he_uniform", batch_norm=True)
    block.summary((8, 8, 8, 2))
