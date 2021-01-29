import tensorflow as tf
from enum import Enum

class Optimiser(Enum):
    ADAM = tf.keras.optimizers.Adam
    SGD = tf.keras.optimizers.SGD

