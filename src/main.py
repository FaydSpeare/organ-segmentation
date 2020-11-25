import tensorflow as tf
import numpy as np
from tf_utils import CDFNet
import nibabel as nib


def segment(data, labels):

    #tf_data = tf.data.Dataset.from_tensor_slices(data)
    axial = CDFNet(num_filters=64, num_classes=6)
    axial.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    #outputs = list()
    #for i in tf_data.batch(1):
    #    outputs.append(axial(i))
    #outputs = tf.concat(outputs, axis=0)

    axial.fit(x=data, y=labels, batch_size=1, epochs=50)
    preds = axial.predict(data)

    # Save image 3D array as nii
    nii_label = nib.Nifti1Image(preds, affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/preds.nii')





#DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/'
#DATA_FOLDER = '/home/fayd/Data/CHAOS/1'
DATA_FOLDER = '/scratch/itee/uqfspear/organ-segmentation/1'

def load_data(folder, type='InPhase'):

    x = nib.load(folder + f'/{type}.nii').get_fdata().astype(np.float32) / 1000.
    x = np.moveaxis(x, -1, 0)
    if len(x.shape) < 4: x = tf.expand_dims(x, axis=-1)

    y = nib.load(folder + '/ground.nii').get_fdata().astype(np.float32) / 63.
    y = np.moveaxis(y, -1, 0)
    y = tf.expand_dims(y, axis=-1)

    return x, y



def main():
    x, y = load_data(DATA_FOLDER)
    segment(x, y)


if __name__ == '__main__':
    print(tf.__version__)
    main()