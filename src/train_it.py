from tf_utils import TFRecordsManager, misc, CDFNet
from model.solver import SegSolver
import nibabel as nib
import numpy as np
import tensorflow as tf
import math

def main():

    print('ok')

    params = {
        'loss_fn' : 'CE',
        'out_channels' : 5,
        'learning_rate' : 0.001
    }

    base_path, model_path = misc.get_base_path(training=True)
    tfrecords_path = base_path + 'tfrecords/'

    # Load TFRecords
    tfrm = TFRecordsManager()
    dataset = tfrm.load_datasets_without_batching(tfrecords_path)

    # Create Padded Batches
    padding_shapes = {'X': (304, 304, 2), 'Y': (304, 304)}
    for mode in dataset:
        dataset[mode] = dataset[mode].padded_batch(1, padded_shapes=padding_shapes)

    network = CDFNet(num_filters=64, num_classes=5)
    solver = SegSolver(model_path, params, network)

    for epoch in range(1000):

        for mode in dataset:
            solver.run_epoch(dataset[mode], mode, epoch)

        train_loss = solver.metrics["train"]["loss"]["value"]
        val_loss = solver.metrics["val"]["loss"]["value"]
        print(f'BestTrainLoss:[{train_loss} BestValLoss:[{val_loss}] EST:[{solver.early_stopping_tick}]', flush=True)
        if solver.early_stopping_tick > 10:
            break

    test = load_data(['3'])
    predict(network, test[0])

    # Save model
    # network.save(base_path + '/model')


def predict(model, data):

    base_path = misc.get_base_path(training=False)

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
    nii_label.to_filename(base_path + '/fixed_preds.nii')

    nii_label = nib.Nifti1Image(output, affine=np.eye(4))
    nii_label.to_filename(base_path + '/preds.nii')


def load_data(folders, type='Combined'):

    base_path = misc.get_base_path(training=False)

    x, y = list(), list()
    for folder in folders:

        a = nib.load(base_path + f'/{folder}/{type}.nii').get_fdata().astype(np.float32) / 1000.
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