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

    output_coronal = np.swapaxes(output_coronal, 0, 1)
    output_sagittal = np.swapaxes(output_sagittal, 0, 2)

    output = np.concatenate([output_axial, output_coronal, output_sagittal], axis=-1)
    output = tf.expand_dims(output, axis=0)

    vab = ViewAggregationBlock(num_filters=30, num_classes=2)
    output = vab.call(output)

    print(output.shape)




DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/'

def main():
    block = tf_utils.CDFNet(num_filters=64, num_classes=3)
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

    data = np.zeros((16, 16, 16, 1))
    label = np.zeros((16, 16, 16, 3))

    segment(data)

    #output = block.call(data)
    #error = tf.keras.losses.mse(label, output)
    #print(error)


if __name__ == '__main__':
    main()