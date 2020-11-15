import nibabel as nib
import tensorflow as tf
import numpy as np

from tf_utils.nets.cdfnet import CDFNet


DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/'

def main():
    block = CDFNet(num_filters=64, num_classes=44)

    data = nib.load(DATA_FOLDER + 'CT/1/4_.nii').get_data() / 1000.
    data = np.moveaxis(data, -1, 0)
    data = tf.expand_dims(data[0], axis=0)
    data = tf.expand_dims(data, axis=-1)

    label = nib.load(DATA_FOLDER + 'CT/1/ground.nii').get_data().astype(np.float32)
    label = np.moveaxis(label, -1, 0)
    label = tf.expand_dims(label[0], axis=0)
    label = tf.expand_dims(label, axis=-1)

    output = block.call(data)
    error = tf.keras.losses.mse(label, output)
    print(error)


if __name__ == '__main__':
    main()