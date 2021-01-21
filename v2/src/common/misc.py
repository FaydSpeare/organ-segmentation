import os
from pathlib import Path
import datetime
import pickle
import json
import tensorflow as tf


def get_base_path():
    return str(Path(__file__).parent.parent.parent)


def get_checkpoint_path():
    path = get_base_path() + '/checkpoints'
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def new_checkpoint_path(prefix=""):
    checkpoint_path = get_checkpoint_path() + "/" + str(prefix) + "ckp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    os.mkdir(checkpoint_path)
    return checkpoint_path


def get_tfrecords_path():
    return get_base_path() + "/tfrecords"


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(path, array):
    with open(path, 'w') as f:
        json.dump(array, f)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def get_argmax_prediction(logits):
    probs = tf.nn.softmax(logits)
    predictions = tf.math.argmax(probs, axis=-1)
    return tf.cast(predictions[..., tf.newaxis], tf.float32)
