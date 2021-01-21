import tensorflow as tf


class Optimisers:

    @staticmethod
    def get_optimiser(params):
        name, lr = params['optimiser'], params['learning_rate']
        if name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=lr)

