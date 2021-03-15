import tensorflow as tf

from model.network.cdfdecoder import CDFDecoder
from model.network.cdfencoder import CDFEncoder


class Network(tf.keras.models.Model):

    def __init__(self, num_filters=64, num_classes=5):
        super(Network, self).__init__()

        self.base_encoder = CDFEncoder(num_filters=num_filters)
        self.base_decoder = CDFDecoder(num_filters=num_filters, num_classes=num_classes)

        self.imitating_encoder = CDFEncoder(num_filters=num_filters)

        self.label_encoder = CDFEncoder(num_filters=num_filters)
        self.label_decoder = CDFDecoder(num_filters=num_filters, num_classes=num_classes)


    def call(self, inputs, training=None, mask=None):

        assert len(inputs) == 2
        x = inputs[0]
        y = inputs[1]

        base_en_out, skips = self.base_encoder(x)
        base_de_out = self.base_decoder([base_en_out] + skips)

        im_en_out, _ = self.imitating_encoder(x)
        la_en_out, _ = self.label_encoder(y)

        im_de_out = self.label_decoder([im_en_out] + skips)
        la_de_out = self.label_decoder([la_en_out] + skips)

        return [base_de_out, im_en_out, la_de_out, im_de_out], [la_en_out, im_en_out]


    def get_config(self):
        pass


    def plot_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        tf.keras.utils.plot_model(model, to_file='CDFNet.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = Network(num_filters=64, num_classes=44)
    block.plot_summary((64, 64, 64))



