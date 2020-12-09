import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import nibabel as nib

from tf_utils.tfrecords import TFRecordsManager
from tf_utils import misc
from tf_utils import CDFNet


def train():

    base_path = misc.get_base_path(training=False)
    tfrecords_path = base_path + 'tfrecords/'

    tfrm = TFRecordsManager()

    dataset = tfrm.load_datasets_without_batching(tfrecords_path)['train']
    dataset = dataset.padded_batch(20, padded_shapes={'X' : (304, 304, 2), 'Y' : (304, 304)})
    dataset = dataset.map(lambda x: (x['X'], x['Y']))


    axial_model = CDFNet(num_filters=64, num_classes=6)
    axial_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stopping_callback = EarlyStopping(restore_best_weights=True, patience=5)

    # Train
    axial_model.fit(dataset, epochs=100,
                    callbacks=[tensorboard_callback, early_stopping_callback])

    # Save model
    axial_model.save(DATA_FOLDER + '/model')




if __name__ == '__main__':
    train()