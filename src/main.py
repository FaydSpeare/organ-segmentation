import tensorflow as tf
import numpy as np
from tf_utils import CDFNet
import nibabel as nib


def segment(data):

    tf_data = tf.data.Dataset.from_tensor_slices(data)
    axial = CDFNet(num_filters=64, num_classes=2)

    outputs = list()
    for i in tf_data.batch(1):
        outputs.append(axial(i))

    outputs = tf.concat(outputs, axis=0)
    print(outputs.shape)

    axial.fit()





#DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/'
DATA_FOLDER = '/scratch/itee/uqfspear/organ-segmentation/'

def main():


    data = nib.load(DATA_FOLDER + '4_.nii').get_fdata().astype(np.float32) / 1000.
    data = np.moveaxis(data, -1, 0)
    data = tf.expand_dims(data, axis=-1)

    #label = nib.load(DATA_FOLDER + 'CT/1/ground.nii').get_data().astype(np.float32) / 255.
    #label = np.moveaxis(label, -1, 0)
    #label = tf.expand_dims(label, axis=0)
    #label = tf.expand_dims(label, axis=-1)


    #data = np.zeros((16, 64, 64, 1)).astype(np.float32)
    #label = np.zeros((16, 64, 64, 3)).astype(np.float32)

    segment(data)


if __name__ == '__main__':
    print(tf.__version__)
    main()