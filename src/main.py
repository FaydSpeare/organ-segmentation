import tensorflow as tf
import numpy as np
from tf_utils import CDFNet
import nibabel as nib

#DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/'
#DATA_FOLDER = '/home/fayd/Data/CHAOS'
DATA_FOLDER = '/scratch/itee/uqfspear/organ-segmentation'

def segment(data, labels):

    axial_model = CDFNet(num_filters=64, num_classes=6)
    axial_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    axial_model.fit(x=data, y=labels, batch_size=1, epochs=50)
    preds = axial_model.predict(data)

    # Save model
    axial_model.save(DATA_FOLDER + '/model')

    # Save image 3D array as nii
    nii_label = nib.Nifti1Image(preds, affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/preds.nii')


def load_data(folders, type='InPhase'):

    x, y = list(), list()
    for folder in folders:

        a = nib.load(DATA_FOLDER + f'/{folder}/{type}.nii').get_fdata().astype(np.float32) / 1000.
        a = np.moveaxis(a, -1, 0)
        if len(a.shape) < 4: a = tf.expand_dims(a, axis=-1)
        x.append(a)

        b = nib.load(DATA_FOLDER + f'/{folder}/ground.nii').get_fdata().astype(np.float32) / 63.
        b = np.moveaxis(b, -1, 0)
        b = tf.expand_dims(b, axis=-1)
        y.append(b)

    return tf.concat(x, axis=0), tf.concat(y, axis=0)


if __name__ == '__main__':
    print(tf.__version__)
    x, y = load_data(['2', '3'])
    segment(x, y)