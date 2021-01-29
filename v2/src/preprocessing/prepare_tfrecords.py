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


def create_tfrecords(volume_name, records_name, save_data=True):

    # Create new folders for data
    tfrecord_path = misc.get_tfrecords_path() + f'/{records_name}'
    data_path = misc.get_data_path() + f'/{records_name}'
    misc.mkdir(tfrecord_path)
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
    paths = [tfrecord_path]
    if save_data: paths.append(data_path)
    for idx, path in enumerate(paths):
        for view in ['axial', 'sagittal', 'coronal']:
            view_path = path + f'/{view}/'
            misc.mkdir(view_path)
            if idx == 0: # Only the tfrecords folder contains these
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

        AX, SAG, COR = 32, 288, 288

        # Axial view
        axial_data = np.moveaxis(data, 2, 0)
        axial_data = tf.image.resize_with_crop_or_pad(axial_data, SAG, COR)
        if save_data: misc.save_nii(axial_data, data_path + f'/axial/{folder}/{folder}-data')
        axial_data = tf.cast(axial_data, dtype=tf.float32)

        axial_label = np.moveaxis(label, 2, 0)
        axial_label = tf.image.resize_with_crop_or_pad(axial_label, SAG, COR)
        axial_label = tf.squeeze(axial_label, axis=-1)
        if save_data: misc.save_nii(axial_label, data_path + f'/axial/{folder}/{folder}-label')
        axial_label = tf.cast(axial_label, dtype=tf.float32) / 63.0

        print(f'Axial: data shape: {axial_data.shape} ~ label shape: {axial_label.shape}')
        sample = [{'X': axial_data[i], 'Y': axial_label[i]} for i in range(len(axial_data))]
        tfrm.save_record(tfrecord_path + f'/axial/{data_purpose}/{folder}', sample)

        # Sagittal view
        sagittal_data = np.moveaxis(data, 1, 0)
        sagittal_data = tf.image.resize_with_crop_or_pad(sagittal_data, COR, AX)
        sagittal_data = tf.cast(tf.image.resize(sagittal_data, [COR, COR//2]), dtype=tf.float64)
        if save_data: misc.save_nii(sagittal_data, data_path + f'/sagittal/{folder}/{folder}-label')
        sagittal_data = tf.cast(sagittal_data, dtype=tf.float32)

        sagittal_label = np.moveaxis(label, 1, 0)
        sagittal_label = tf.image.resize_with_crop_or_pad(sagittal_label, COR, AX)
        sagittal_label = resize_label(sagittal_label, [COR, COR//2])
        sagittal_label = tf.squeeze(sagittal_label, axis=-1)
        if save_data: misc.save_nii(sagittal_label, data_path + f'/sagittal/{folder}/{folder}-label')
        sagittal_label = tf.cast(sagittal_label, dtype=tf.float32) / 63.0

        print(f'Sagittal: data shape: {sagittal_data.shape} ~ label shape: {sagittal_label.shape}')
        sample = [{'X': sagittal_data[i], 'Y': sagittal_label[i]} for i in range(len(sagittal_data))]
        tfrm.save_record(tfrecord_path + f'/sagittal/{data_purpose}/{folder}', sample)

        # Coronal view
        coronal_data = tf.image.resize_with_crop_or_pad(data, SAG, AX)
        coronal_data = tf.cast(tf.image.resize(coronal_data, [SAG, SAG // 2]), dtype=tf.float64)
        if save_data: misc.save_nii(coronal_data, data_path + f'/coronal/{folder}/{folder}-data')
        coronal_data = tf.cast(coronal_data, dtype=tf.float32)

        coronal_label = tf.image.resize_with_crop_or_pad(label, SAG, AX)
        coronal_label = resize_label(coronal_label, [SAG, SAG // 2])
        coronal_label = tf.squeeze(coronal_label, axis=-1)
        if save_data: misc.save_nii(coronal_label, data_path + f'/coronal/{folder}/{folder}-label')
        coronal_label = tf.cast(coronal_label, dtype=tf.float32) / 63.0

        print(f'Coronal: data shape: {coronal_data.shape} ~ label shape: {coronal_label.shape}')
        sample = [{'X': coronal_data[i], 'Y': coronal_label[i]} for i in range(len(coronal_data))]
        tfrm.save_record(tfrecord_path + f'/coronal/{data_purpose}/{folder}', sample)

        print('\n')



if __name__ == '__main__':
    create_tfrecords('Normalized.nii', '3z-normal')