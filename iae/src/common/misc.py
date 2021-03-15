import os
from pathlib import Path
import datetime
import pickle
import json
import tensorflow as tf
import numpy as np
import nibabel as nib


def get_base_path():
    return str(Path(__file__).parent.parent.parent)


def get_checkpoint_path():
    return mkdir(get_base_path() + '/checkpoints')


def new_checkpoint_path(prefix="_", tfr="_"):
    date = datetime.datetime.now().strftime("%b-%d-%H%M%S")
    tfr = tfr.replace('/', '_')
    checkpoint_path = get_checkpoint_path() + f"/({prefix})-({tfr})-({date})/"
    os.mkdir(checkpoint_path)
    return checkpoint_path


def get_tfrecords_path():
    return mkdir(get_base_path() + "/tfrecords")


def get_chaos_path():
    return get_base_path() + '/../CHAOS'


def get_data_path():
    return mkdir(get_base_path() + "/data")


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

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


def save_nii(volume, path, affine=np.eye(4), header=None):
    nib.Nifti1Image(volume, affine=affine, header=header).to_filename(path)
