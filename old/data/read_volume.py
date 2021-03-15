import nibabel as nib
from tf_utils import CDFNet, misc
import numpy as np
import tensorflow as tf
import math

VOLUME_PATH = '/home/fayd/Data/Unlabelled/data/sub-10010_{}.nii'

if __name__ == '__main__':

    for i in [16]:
        og_data = nib.load(VOLUME_PATH.format(i)).get_fdata()
        base_path, _ = misc.get_base_path(training=True)

        nii_label = nib.Nifti1Image(tf.pad(og_data, paddings=tf.constant([[0, 0, ], [3, 3, ], [0, 0, ]])), affine=np.eye(4))
        nii_label.to_filename(base_path + f'/data_{i}.nii')

        data = np.moveaxis(og_data, -1, 0).astype(np.float32)
        data = data / float(np.max(data) / 2)
        data = tf.expand_dims(data, axis=-1)
        data = tf.concat([data, data], axis=-1)
        data_combined = tf.pad(data, paddings=tf.constant([[0, 0,], [0, 0,], [3, 3,], [0, 0,]]))
        network = CDFNet(num_filters=64, num_classes=5)
        network.load_weights(base_path + 'dice_model/model_weights')

        xs = []
        for x in tf.data.Dataset.from_tensor_slices(data_combined).batch(10):
            print(len(xs))
            out = network.predict(x)
            xs.append(out)

        output = np.concatenate(xs, axis=0)

        #output = np.array(xs)
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
        nii_label.to_filename(base_path + f'/preds_{i}.nii')

        # Save image 3D array as nii
        og_data = tf.pad(og_data, paddings=tf.constant([[0, 0,], [3, 3,], [0, 0,]]))
        nii_label = nib.Nifti1Image(og_data, affine=np.eye(4))
        nii_label.to_filename(base_path + f'/data_{i}.nii')
        print()

