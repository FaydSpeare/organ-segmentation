import os
import nibabel as nib
import numpy as np
import math
import tensorflow as tf

from common.tfrecords import TFRecordsManager
from common import misc


DATA_FOLDER = '/home/fayd/Fayd/Projects/organ-segmentation/CHAOS'


def resize_label(label, size, alpha=63.0):
    return tf.round(tf.image.resize(label, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) / alpha) * alpha


def create_tfrecords(volume_name, records_name, save_data=True, save_record=True):

    # Create new folders for data
    tfrecord_path = misc.get_tfrecords_path() + f'/{records_name}'
    data_path = misc.get_data_path() + f'/{records_name}'
    if save_record: misc.mkdir(tfrecord_path)
    if save_data: misc.mkdir(data_path)

    tfrm = TFRecordsManager()
    total_samples = len(os.listdir('/home/fayd/Data/CHAOS'))
    split = math.floor(0.8 * total_samples)
    params = {
        'data_purposes': ['train', 'val'],
        'data_keys': {
            'X': 'float32',
            'Y': 'float32'
        }
    }

    # Create necessary folders
    paths = []
    if save_record: paths.append(tfrecord_path)
    if save_data: paths.append(data_path)
    for idx, path in enumerate(paths):
        for view in ['axial', 'sagittal', 'coronal']:
            view_path = path + f'/{view}/'
            misc.mkdir(view_path)
            if path == tfrecord_path: # Only the tfrecords folder contains these
                misc.save_json(view_path + 'params.json', params)
                for data_purpose in ['train', 'val']:
                    misc.mkdir(view_path + data_purpose)

    # Create records
    for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
        print(f'Creating TFRecord for folder: [{folder}]')
        data_purpose = 'train' if idx <= split else 'val'

        if save_data:
            for view in ['axial', 'sagittal', 'coronal']:
                misc.mkdir(f'{data_path}/{view}/{folder}')

        # Load data and labels
        data = nib.load(f'{DATA_FOLDER}/{folder}/{volume_name}').get_fdata()
        label = nib.load(f'{DATA_FOLDER}/{folder}/ground.nii').get_fdata()
        label = tf.expand_dims(label, axis=-1)

        # RESIZE DATA TO (144, 288, 288)
        AX, SAG, COR = 144, 288, 288
        data = np.moveaxis(data, 2, 0)
        label = np.moveaxis(label, 2, 0)
        data = tf.cast(tf.image.resize(data, [SAG, COR]), dtype=tf.float64)
        label = resize_label(label, [SAG, COR])
        data = np.moveaxis(data, 0, 2)
        label = np.moveaxis(label, 0, 2)
        data = tf.image.resize(data, [SAG, AX]).numpy().astype(np.float32)
        label = resize_label(label, [SAG, AX])
        label = tf.squeeze(label, axis=-1)
        label = label.numpy().astype(np.int8)

        # Axial view
        axial_data = np.moveaxis(data, 2, 0)
        if save_data:
            misc.save_nii(axial_data, data_path + f'/axial/{folder}/{folder}-data')

        axial_label = np.moveaxis(label, 2, 0)
        if save_data: misc.save_nii(axial_label, data_path + f'/axial/{folder}/{folder}-label')
        axial_label = axial_label.astype(np.float32) / 63.0

        print(f'Axial: data shape: {axial_data.shape} ~ label shape: {axial_label.shape}')
        if save_record:
            sample = [{'X': axial_data[i], 'Y': axial_label[i]} for i in range(len(axial_data))]
            tfrm.save_record(tfrecord_path + f'/axial/{data_purpose}/{folder}', sample)

        # Sagittal view
        sagittal_data = np.moveaxis(data, 1, 0)
        if save_data: misc.save_nii(sagittal_data, data_path + f'/sagittal/{folder}/{folder}-data')

        sagittal_label = np.moveaxis(label, 1, 0)
        if save_data: misc.save_nii(sagittal_label, data_path + f'/sagittal/{folder}/{folder}-label')
        sagittal_label = sagittal_label.astype(np.float32) / 63.0
        print(f'Sagittal: data shape: {sagittal_data.shape} ~ label shape: {sagittal_label.shape}')
        if save_record:
            sample = [{'X': sagittal_data[i], 'Y': sagittal_label[i]} for i in range(len(sagittal_data))]
            tfrm.save_record(tfrecord_path + f'/sagittal/{data_purpose}/{folder}', sample)

        # Coronal view
        coronal_data = data
        if save_data: misc.save_nii(coronal_data, data_path + f'/coronal/{folder}/{folder}-data')

        coronal_label = label
        if save_data: misc.save_nii(coronal_label, data_path + f'/coronal/{folder}/{folder}-label')
        coronal_label = coronal_label.astype(np.float32) / 63.0
        print(f'Coronal: data shape: {coronal_data.shape} ~ label shape: {coronal_label.shape}')
        if save_record:
            sample = [{'X': coronal_data[i], 'Y': coronal_label[i]} for i in range(len(coronal_data))]
            tfrm.save_record(tfrecord_path + f'/coronal/{data_purpose}/{folder}', sample)

        print('\n')



if __name__ == '__main__':
    create_tfrecords('Normalized.nii', 'view-agg-data', save_record=False)