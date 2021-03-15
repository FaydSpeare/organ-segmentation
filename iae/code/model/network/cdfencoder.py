import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D

from model.blocks.competitive_dense_block import CompDenseBlock


class CDFEncoder(tf.keras.models.Model):

    def __init__(self, num_filters=64):
        super(CDFEncoder, self).__init__()

        self.cdb_encoder1 = CompDenseBlock(num_filters, is_input_block=True)
        self.max_pool1 = MaxPool2D()

        self.cdb_encoder2 = CompDenseBlock(num_filters)
        self.max_pool2 = MaxPool2D()

        self.cdb_encoder3 = CompDenseBlock(num_filters)
        self.max_pool3 = MaxPool2D()

        self.cdb_encoder4 = CompDenseBlock(num_filters)
        self.max_pool4 = MaxPool2D()

        self.bottleneck = CompDenseBlock(num_filters)


    def call(self, input_tensor, training=None, mask=None):

        x = input_tensor

        encoder1 = self.cdb_encoder1(x, training=training)
        x = self.max_pool1(encoder1)

        encoder2 = self.cdb_encoder2(x, training=training)
        x = self.max_pool2(encoder2)

        encoder3 = self.cdb_encoder3(x, training=training)
        x = self.max_pool3(encoder3)

        encoder4 = self.cdb_encoder3(x, training=training)
        x = self.max_pool3(encoder4)

        x = self.bottleneck(x, training=training)

        return x, [encoder1, encoder2, encoder3, encoder4]

    def get_config(self):
        pass


    def plot_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        # tf.keras.utils.plot_model(model, to_file='CDFNet.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = CDFEncoder(num_filters=64)
    block.plot_summary((64, 64, 64))