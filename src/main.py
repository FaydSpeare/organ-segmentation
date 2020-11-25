import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
from tf_utils import CDFNet
import nibabel as nib

#DATA_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/'
#DATA_FOLDER = '/home/fayd/Data/CHAOS'
DATA_FOLDER = '/scratch/itee/uqfspear/organ-segmentation'

def segment(train, val):

    x_train, y_train = train
    x_val, y_val = val

    axial_model = CDFNet(num_filters=64, num_classes=6)
    axial_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stopping_callback = EarlyStopping(restore_best_weights=True, patience=5)

    # Train
    axial_model.fit(x=x_train, y=y_train, batch_size=1, epochs=100, validation_data=(x_val, y_val),
                    callbacks=[tensorboard_callback, early_stopping_callback])

    # Save model
    axial_model.save(DATA_FOLDER + '/model')
    return axial_model


def predict(model, data):

    pred = model.predict(data)

    # Save image 3D array as nii
    nii_label = nib.Nifti1Image(pred, affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/pred.nii')


def load_data(folders, type='InPhase'):

    x, y = list(), list()
    for folder in folders:

        a = nib.load(DATA_FOLDER + f'/{folder}/{type}.nii').get_fdata().astype(np.float32) / 1000.
        a = np.moveaxis(a, -1, 0)
        print(a.shape)
        if len(a.shape) < 4: a = tf.expand_dims(a, axis=-1)
        x.append(a)

        b = nib.load(DATA_FOLDER + f'/{folder}/ground.nii').get_fdata().astype(np.float32) / 63.
        b = np.moveaxis(b, -1, 0)
        b = tf.expand_dims(b, axis=-1)
        y.append(b)

    return tf.concat(x, axis=0), tf.concat(y, axis=0)


if __name__ == '__main__':
    print(tf.__version__)
    train_data = load_data(['2'])
    val_data = load_data(['3'])
    model = segment(train_data, val_data)
    predict(model, val_data[0])
