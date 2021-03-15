import tensorflow as tf
from enum import Enum

class Optimiser(Enum):
    ADAM = tf.keras.optimizers.Adam
    SGD = tf.keras.optimizers.SGD

    @staticmethod
    def optimiser(o):
        if o == Optimiser.ADAM:
            return tf.keras.optimizers.Adam
        elif o == Optimiser.SGD:
            return tf.keras.optimizers.SGD


if __name__ == '__main__':
    a = Optimiser.optimiser(Optimiser.ADAM)(learning_rate=0.001)

