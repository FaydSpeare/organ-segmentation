import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, Conv2D, Softmax

from tf_utils.blocks.competitive_unpool_block import CompUnpoolBlock
from tf_utils.blocks.competitive_dense_block import CompDenseBlock

class CDFNet(tf.keras.models.Model):

    def __init__(self, num_filters=64, num_classes=44):
        super(CDFNet, self).__init__()

        self.cdb_encoder1 = CompDenseBlock(num_filters, is_input_block=True)
        self.max_pool1 = MaxPool2D()

        self.cdb_encoder2 = CompDenseBlock(num_filters)
        self.max_pool2 = MaxPool2D()

        self.cdb_encoder3 = CompDenseBlock(num_filters)
        self.max_pool3 = MaxPool2D()

        self.cdb_encoder4 = CompDenseBlock(num_filters)
        self.max_pool4 = MaxPool2D()

        self.bottleneck = CompDenseBlock(num_filters)

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

        pass

    def call(self, input_tensor, training=None, mask=None):

        x = input_tensor

        encoder1 = self.cdb_encoder1(x)
        x = self.max_pool1(encoder1)

        encoder2 = self.cdb_encoder2(x)
        x = self.max_pool2(encoder2)

        encoder3 = self.cdb_encoder3(x)
        x = self.max_pool3(encoder3)

        encoder4 = self.cdb_encoder3(x)
        x = self.max_pool3(encoder4)

        x = self.bottleneck(x)

        x = self.up_sample4(x)
        x = self.cub4([x, encoder4])
        x = self.cdb_decoder4(x)

        x = self.up_sample3(x)
        x = self.cub3([x, encoder3])
        x = self.cdb_decoder3(x)

        x = self.up_sample2(x)
        x = self.cub2([x, encoder2])
        x = self.cdb_decoder2(x)

        x = self.up_sample1(x)
        x = self.cub1([x, encoder1])
        x = self.cdb_decoder1(x)

        x = self.final_conv(x)
        x = self.softmax(x)

        return x

    def get_config(self):
        pass


    def plot_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='CDFNet.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = CDFNet(num_filters=64, num_classes=44)
    block.plot_summary((64, 64, 64))



