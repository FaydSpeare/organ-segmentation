import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Softmax

from model.blocks.competitive_dense_block import CompDenseBlock
from model.blocks.competitive_unpool_block import CompUnpoolBlock


class CDFDecoder(tf.keras.models.Model):

    def __init__(self, num_filters=64, num_classes=5):
        super(CDFDecoder, self).__init__()

        self.up_sample4 = Conv2DTranspose(num_filters, kernel_size=2, strides=(2, 2), padding='same')
        self.cub4 = CompUnpoolBlock(num_filters)
        self.cdb_decoder4 = CompDenseBlock(num_filters)

        self.up_sample3 = Conv2DTranspose(num_filters, kernel_size=2, strides=(2, 2), padding='same')
        self.cub3 = CompUnpoolBlock(num_filters)
        self.cdb_decoder3 = CompDenseBlock(num_filters)

        self.up_sample2 = Conv2DTranspose(num_filters, kernel_size=2, strides=(2, 2), padding='same')
        self.cub2 = CompUnpoolBlock(num_filters)
        self.cdb_decoder2 = CompDenseBlock(num_filters)

        self.up_sample1 = Conv2DTranspose(num_filters, kernel_size=2, strides=(2, 2), padding='same')
        self.cub1 = CompUnpoolBlock(num_filters)
        self.cdb_decoder1 = CompDenseBlock(num_filters)

        self.final_conv = Conv2D(num_classes, kernel_size=1)
        self.softmax = Softmax()


    def call(self, inputs, training=None, mask=None):

        x, encoder_outputs = inputs[0], inputs[1:]

        x = self.up_sample4(x)
        x = self.cub4([x, encoder_outputs[3]], training=training)
        x = self.cdb_decoder4(x, training=training)

        x = self.up_sample3(x)
        x = self.cub3([x, encoder_outputs[2]], training=training)
        x = self.cdb_decoder3(x, training=training)

        x = self.up_sample2(x)
        x = self.cub2([x, encoder_outputs[1]], training=training)
        x = self.cdb_decoder2(x, training=training)

        x = self.up_sample1(x)
        x = self.cub1([x, encoder_outputs[0]], training=training)
        x = self.cdb_decoder1(x, training=training)

        x = self.final_conv(x)
        x = self.softmax(x)

        return x

    def get_config(self):
        pass




