import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import nibabel as nib
import math

from tf_utils.tfrecords import TFRecordsManager
from tf_utils import misc
from tf_utils import CDFNet


def train():

    base_path = misc.get_base_path(training=False)
    tfrecords_path = base_path + 'tfrecords/'

    # Load TFRecords
    tfrm = TFRecordsManager()
    dataset = tfrm.load_datasets_without_batching(tfrecords_path)

    # Prep training data
    train_dataset = dataset['train']
    train_dataset = train_dataset.padded_batch(5, padded_shapes={'X' : (304, 304, 2), 'Y' : (304, 304)})
    train_dataset = train_dataset.map(lambda x: (x['X'], x['Y']))

    # Prep validation data
    val_dataset = dataset['val']
    val_dataset = val_dataset.padded_batch(5, padded_shapes={'X' : (304, 304, 2), 'Y' : (304, 304)})
    val_dataset = val_dataset.map(lambda x: (x['X'], x['Y']))

    # Create model
    model = CDFNet(num_filters=64, num_classes=5)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Fit model
    tensorboard_callback = TensorBoard(log_dir="./logs")
    early_stopping_callback = EarlyStopping(restore_best_weights=True, patience=3)
    model.fit(train_dataset, epochs=15, callbacks=[tensorboard_callback, early_stopping_callback], validation_data=val_dataset)

    # Save model
    model.save(base_path + '/model')

    test = load_data(['2'])
    predict(model, test[0])



def predict(model, data):

    base_path = misc.get_base_path(training=False)

    output = model.predict(data)

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
    nii_label.to_filename(base_path + '/fixed_preds.nii')

    nii_label = nib.Nifti1Image(output, affine=np.eye(4))
    nii_label.to_filename(base_path + '/preds.nii')


def load_data(folders, type='Combined'):

    base_path = misc.get_base_path(training=False)

    x, y = list(), list()
    for folder in folders:

        a = nib.load(base_path + f'/{folder}/{type}.nii').get_fdata().astype(np.float32) / 1000.
        a = np.moveaxis(a, -2, 0)
        print(a.shape)
        if len(a.shape) < 4: a = tf.expand_dims(a, axis=-1)
        x.append(a)

        b = nib.load(base_path + f'/{folder}/ground.nii').get_fdata().astype(np.float32) / 63.
        b = np.moveaxis(b, -1, 0)
        b = tf.expand_dims(b, axis=-1)
        y.append(b)

    return tf.concat(x, axis=0), tf.concat(y, axis=0)


if __name__ == '__main__':
    train()


'''
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

for batch in dataset: # (batch, X, Y)

    with tf.GradientTape() as tape:
        logits = model(batch[0])
        loss = loss_fn(batch[1], logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
'''