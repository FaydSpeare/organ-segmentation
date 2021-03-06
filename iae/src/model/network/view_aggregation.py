import tensorflow as tf
from tensorflow.keras.layers import Conv3D, BatchNormalization, Softmax, ReLU

class ViewAggregation(tf.keras.models.Model):

    def __init__(self, num_filters=30, num_classes=5):
        super(ViewAggregation, self).__init__()

        self.conv1 = Conv3D(filters=num_filters, kernel_size=3, padding='same')
        self.batch_norm1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv3D(filters=num_classes, kernel_size=1, padding='same')
        self.softmax = Softmax()


    def call(self, input_tensor, training=None, mask=None):

        x = input_tensor
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.softmax(x)

        return x

    def get_config(self):
        pass

    def plot_summary(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=False))
        #tf.keras.utils.plot_model(model, to_file='VAB.png', show_shapes=True, expand_nested=True)
        model.summary(line_length=200)


if __name__ == '__main__':
    block = ViewAggregation(num_filters=30, num_classes=5)
    block.plot_summary((32, 32, 32, 15))
