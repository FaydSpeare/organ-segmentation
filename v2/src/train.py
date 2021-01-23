import numpy as np
import nibabel as nib
import tensorflow as tf
import math

from common.tfrecords import TFRecordsManager
from model.solver import Solver
from model.network import CDFNet
from common import misc


def main():

    params = {
        'tfrecords' : 'sagittal',
        'loss_fn' : 'CCE',
        'out_channels' : 5,
        'learning_rate' : 0.001,
        'optimiser' : 'adam',
        'modes' : ['train', 'val'],
        'batch_size' : 30
    }

    # Create new folder for training
    params['path'] = misc.new_checkpoint_path(params['tfrecords'])

    # Load TFRecords
    tfrm = TFRecordsManager()
    dataset = tfrm.load_datasets(misc.get_tfrecords_path() + f"/{params['tfrecords']}/", params['batch_size'])

    network = CDFNet(num_filters=64, num_classes=5)
    solver = Solver(network, params)
    epoch_metrics = dict()

    for epoch in range(1000):

        for mode in dataset:
            epoch_metrics[mode] = solver.run_epoch(dataset[mode], mode)

        best_val_loss = solver.best_val_loss
        val_loss = epoch_metrics['val']['loss']
        print(f'ValLoss:[{val_loss}] BestValLoss:[{best_val_loss}] EST:[{solver.early_stopping_tick}]', flush=True)
        if solver.early_stopping_tick > 10:
            break

    test = load_data(['3'])
    predict(params['path'], network, test[0])


def predict(path, model, data):

    output = model.predict(data)

    labels = np.swapaxes(output, 0, 2)
    labels = np.swapaxes(labels, 0, 1)

    # Collapse the probabilities into a one hot encoding
    labels = np.eye(labels.shape[-1])[labels.argmax(axis=-1)]

    # Multiply out and sum
    num_classes = labels.shape[-1]
    labels *= [i * math.floor(255. / num_classes) for i in range(num_classes)]
    labels = np.sum(labels, axis=-1)

    # Save image 3D array as nii
    nii_label = nib.Nifti1Image(labels, affine=np.eye(4))
    nii_label.to_filename(path + '/seg.nii')

    nii_label = nib.Nifti1Image(output, affine=np.eye(4))
    nii_label.to_filename(path + '/pred.nii')


def load_data(folders, type='Combined'):
    base_path = misc.get_base_path()

    x, y = list(), list()
    for folder in folders:

        a = nib.load(base_path + f'/{folder}/{type}.nii').get_fdata().astype(np.float32)
        a = a / float(np.max(a) / 2)
        a = np.moveaxis(a, -2, 0)
        print(a.shape)
        if len(a.shape) < 4: a = tf.expand_dims(a, axis=-1)
        x.append(a)

        b = nib.load(base_path + f'/{folder}/ground.nii').get_fdata().astype(np.float32) / 63.
        b = np.moveaxis(b, -1, 0)
        b = tf.expand_dims(b, axis=-1)
        y.append(b)

    return tf.concat(x, axis=0), tf.concat(y, axis=0)

if __name__ == '__main__':
    main()