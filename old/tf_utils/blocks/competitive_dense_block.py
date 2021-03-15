import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, BatchNormalization
from tf_utils.layers.maxout import Maxout

class CompDenseBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, is_input_block=False):
        super(CompDenseBlock, self).__init__()
        self.is_input_block = is_input_block

        self.batch_norm11 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv1 = Conv2D(num_filters, kernel_size=5, padding='same')
        self.batch_norm12 = BatchNormalization()

        self.concat = Concatenate()
        self.max_out1 = Maxout()

        self.batch_norm21 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv2D(num_filters, kernel_size=5, padding='same')
        self.batch_norm22 = BatchNormalization()

        self.max_out2 = Maxout()

        self.relu3 = ReLU()
        self.conv3 = Conv2D(num_filters, kernel_size=1, padding='same')

    def call(self, input_tensor, training=None, mask=None):

        x = input_tensor
        concat = x

        x = self.batch_norm11(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.batch_norm12(x)

        if self.is_input_block:
            concat = x
            x = self.concat([x, concat])

        else:
            x = self.max_out1([x, concat])
            concat = x

        x = self.batch_norm21(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm22(x)

        x = self.max_out2([x, concat])

        x = self.relu3(x)
        x = self.conv3(x)

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
    block.plot_summary((16, 16, 64))
