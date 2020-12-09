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
    model = CDFNet(num_filters=64, num_classes=6)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Fit model
    tensorboard_callback = TensorBoard(log_dir="./logs")
    model.fit(train_dataset, epochs=10, callbacks=[tensorboard_callback], validation_data=val_dataset)

    # Save model
    model.save(base_path + '/model')


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