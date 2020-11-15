import tensorflow as tf
import numpy as np
from tf_utils import CDFNet, ViewAggregationBlock




def segment(data):

    data_axial = data
    data_coronal = np.swapaxes(data, 0, 1)
    data_sagittal = np.swapaxes(data, 0, 2)

    axial = CDFNet(num_filters=64, num_classes=2)
    coronal = CDFNet(num_filters=64, num_classes=2)
    sagittal = CDFNet(num_filters=64, num_classes=2)

    output_axial = axial(data_axial)
    output_coronal = coronal(data_coronal)
    output_sagittal = sagittal(data_sagittal)

    print('Axial:', output_axial.shape)
    print('Coronal:', output_coronal.shape)
    print('Sagittal:', output_sagittal.shape)

    output_coronal = tf.transpose(output_coronal, [1, 0, 2, 3])
    output_sagittal = tf.transpose(output_sagittal, [2, 1, 0, 3])

    output = np.concatenate([output_axial, output_coronal, output_sagittal], axis=-1)
    output = tf.expand_dims(output, axis=0)

    vab = ViewAggregationBlock(num_filters=30, num_classes=2)
    output = vab.call(output)

    print(output.shape)




DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/'

def main():

    '''
    data = nib.load(DATA_FOLDER + 'CT/1/4_.nii').get_data().astype(np.float32) / 1000.
    data = np.moveaxis(data, -1, 0)
    data = tf.expand_dims(data, axis=0)
    data = tf.expand_dims(data, axis=-1)

    label = nib.load(DATA_FOLDER + 'CT/1/ground.nii').get_data().astype(np.float32) / 255.
    label = np.moveaxis(label, -1, 0)
    label = tf.expand_dims(label, axis=0)
    label = tf.expand_dims(label, axis=-1)
    '''

    data = np.zeros((96, 32, 96, 1)).astype(np.float32)
    label = np.zeros((96, 32, 96, 3)).astype(np.float32)

    segment(data)

    #output = block.call(data)
    #error = tf.keras.losses.mse(label, output)
    #print(error)


if __name__ == '__main__':
    print(tf.__version__)
    main()