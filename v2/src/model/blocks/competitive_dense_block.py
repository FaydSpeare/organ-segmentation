import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, BatchNormalization, Dropout
from model.blocks.maxout import Maxout


class CompDenseBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, is_input_block=False):
        super(CompDenseBlock, self).__init__()
        self.is_input_block = is_input_block

        self.conv1 = Conv2D(num_filters, kernel_size=5, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.relu1 = ReLU()

        self.concat = Concatenate(axis=-1)
        self.max_out1 = Maxout()

        self.conv2 = Conv2D(num_filters, kernel_size=5, padding='same')
        self.batch_norm2 = BatchNormalization()
        self.relu2 = ReLU()

        self.max_out2 = Maxout()

        self.conv3 = Conv2D(num_filters, kernel_size=1, padding='same')
        self.batch_norm3 = BatchNormalization()
        self.relu3 = ReLU()

    def call(self, input_tensor, training=None):

        x = input_tensor
        concat = x

        x = self.conv1(x)
        x = self.batch_norm11(x, training=training)
        x = self.relu1(x)

        if self.is_input_block:
            concat = x
            x = self.concat([x, concat])

        else:
            x = self.max_out1([x, concat])
            concat = x

        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.relu2(x)

        x = self.max_out2([x, concat])

        x = self.conv3(x)
        x = self.batch_norm3(x, training=training)
        x = self.relu3(x)

        return x

    def get_config(self):
        pass

    def plot_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='CDB.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = CompDenseBlock(64)
    #block.plot_summary((16, 16, 64))
