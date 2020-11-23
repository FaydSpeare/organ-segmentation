import tensorflow as tf
import numpy as np
from tf_utils import CDFNet
import nibabel as nib
import matplotlib.pyplot as plt


def segment(data, labels):

    #tf_data = tf.data.Dataset.from_tensor_slices(data)
    axial = CDFNet(num_filters=64, num_classes=2)
    axial.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    #outputs = list()
    #for i in tf_data.batch(1):
    #    outputs.append(axial(i))
    #outputs = tf.concat(outputs, axis=0)

    axial.fit(x=data, y=labels, batch_size=1, epochs=1)
    preds = axial.predict(data)

    # Save image 3D array as nii
    nii_label = nib.Nifti1Image(preds, affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/preds.nii')





#DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/'
DATA_FOLDER = '/scratch/itee/uqfspear/organ-segmentation/'

def main():

    #data = np.zeros((16, 64, 64, 1)).astype(np.float32)
    #label = np.zeros((16, 64, 64, 3)).astype(np.float32)

    data = nib.load(DATA_FOLDER + '4_.nii').get_fdata().astype(np.float32) / 1000.
    data = np.moveaxis(data, -1, 0)
    data = tf.expand_dims(data, axis=-1)

    label = nib.load(DATA_FOLDER + 'ground.nii').get_fdata().astype(np.float32) / 255.
    label = np.moveaxis(label, -1, 0)

    plt.imshow(label[20])
    plt.show()

    label = tf.expand_dims(label, axis=-1)

    plt.imshow(label.numpy()[20])
    plt.show()

    segment(data, label)


if __name__ == '__main__':
    print(tf.__version__)
    main()