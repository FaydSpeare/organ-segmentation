import os
import nibabel as nib
import numpy as np
import math
import random
import tensorflow as tf

from common.tfrecords import TFRecordsManager
from common import misc



def create_tfrecords():

    tfrecord_path = misc.get_tfrecords_path()
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
    for view in ['axial', 'sagittal', 'coronal']:

        path = tfrecord_path + f'/{view}/'
        if not os.path.isdir(path):
            os.mkdir(path)

        for data_purpose in ['train', 'val']:
            if not os.path.isdir(path + data_purpose):
                os.mkdir(path + data_purpose)

    # Create records
    for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
        print(f'Creating TFRecord for folder: [{folder}]')
        data_purpose = 'train' if idx <= split else 'val'

        data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'
        label_path = f'{DATA_FOLDER}/{folder}/ground.nii'

        # Load data and labels
        data = nib.load(data_path).get_fdata().astype(np.float32)
        data = data / float(np.max(data) / 2)
        label = nib.load(label_path).get_fdata().astype(np.float32) / 63.0
        label = tf.expand_dims(label, axis=-1)

        # Axial view
        axial_data = np.moveaxis(data, 2, 0)
        axial_data = tf.image.resize_with_crop_or_pad(axial_data, 288, 288)

        axial_label = np.moveaxis(label, 2, 0)
        axial_label = tf.image.resize_with_crop_or_pad(axial_label, 288, 288)
        axial_label = tf.squeeze(axial_label, axis=-1)

        print(f'data shape: {axial_data.shape} ~ label shape: {axial_label.shape}')
        sample = [{'X': axial_data[i], 'Y': axial_label[i]} for i in range(len(axial_data))]
        tfrm.save_record(tfrecord_path + f'/axial/{data_purpose}/{folder}', sample)
        print('Create axial record.')

        # Sagittal view

        # Coronal view

        print('\n')


DATA_FOLDER = '/home/fayd/Data/CHAOS'

def main():

    tfrecord_path = misc.get_tfrecords_path()
    tfrm = TFRecordsManager()
    total_samples = len(os.listdir(DATA_FOLDER))
    split = math.floor(0.8 * total_samples)

    for view in ['axial', 'sagittal', 'coronal']:

        path = tfrecord_path + f'/{view}/'
        if not os.path.isdir(path):
            os.mkdir(path)

        misc.save_json(path + 'params.json', {
            'data_purposes': ['train', 'val'],
            'data_keys': {
                'X': 'float32',
                'Y': 'float32'
            }
        })

        for data_purpose in ['train', 'val']:
            if not os.path.isdir(path + data_purpose):
                os.mkdir(path + data_purpose)

        for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
            print(f'Creating TFRecord for folder: [{folder}]')

            data_purpose = 'train' if idx <= split else 'val'

            data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'
            label_path = f'{DATA_FOLDER}/{folder}/ground.nii'

            sample_data = nib.load(data_path).get_fdata().astype(np.float32)
            sample_data = sample_data / float(np.max(sample_data) / 2)
            sample_label = nib.load(label_path).get_fdata().astype(np.float32) / 63.0
            sample_label = tf.expand_dims(sample_label, axis=-1)

            sample_data = tf.image.resize_with_crop_or_pad(sample_data, 288, 64)
            sample_label = tf.image.resize_with_crop_or_pad(sample_label, 288, 64)
            sample_data = np.moveaxis(sample_data, 2, 0)
            sample_label = np.moveaxis(sample_label, 2, 0)
            sample_data = tf.image.resize_with_crop_or_pad(sample_data, 288, 288)
            sample_label = tf.image.resize_with_crop_or_pad(sample_label, 288, 288)
            sample_data = np.moveaxis(sample_data, 0, 2)
            sample_label = np.moveaxis(sample_label, 0, 2)
            sample_label = tf.squeeze(sample_label, axis=-1)

            # Rearrange data
            if view == 'axial':
                sample_data = np.moveaxis(sample_data, 2, 0)
                sample_label = np.moveaxis(sample_label, 2, 0)

            if view == 'sagittal':
                sample_data = np.moveaxis(sample_data, 1, 0)
                sample_label = np.moveaxis(sample_label, 1, 0)


            print(sample_data.shape)
            print()

            sample = [{'X': sample_data[i], 'Y': sample_label[i]} for i in range(len(sample_data))]
            tfrm.save_record(f'{path}/{data_purpose}/{folder}', sample)


def main_patches():

    path = '/home/fayd/Fayd/Projects/organ-segmentation/tfrecords-patches/'

    tfrm = TFRecordsManager()

    misc.save_json(path + 'params.json', {
        'data_purposes': ['train', 'val'],
        'data_keys': {
            'X': 'float32',
            'Y': 'float32'
        }
    })

    for data_purpose in ['train', 'val']:
        if not os.path.isdir(path + data_purpose):
            os.mkdir(path + data_purpose)

    total_samples = len(os.listdir(DATA_FOLDER))
    split = math.floor(0.8 * total_samples)

    for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
        print(f'Creating TFRecord for folder: [{folder}]')

        data_purpose = 'train' if idx <= split else 'val'

        data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'
        label_path = f'{DATA_FOLDER}/{folder}/ground.nii'

        sample_data = nib.load(data_path).get_fdata().astype(np.float32) / 1000.
        sample_label = nib.load(label_path).get_fdata().astype(np.float32) / 63.

        # Rearrange data
        sample_data = np.moveaxis(sample_data, 2, 0)
        sample_label = np.moveaxis(sample_label, 2, 0)

        print(np.sum(np.count_nonzero(sample_label == 0., axis=-1)) / (288 * 288 * 30))
        print(np.sum(np.count_nonzero(sample_label == 1., axis=-1)) / (288 * 288 * 30))
        print(np.sum(np.count_nonzero(sample_label == 2., axis=-1)) / (288 * 288 * 30))
        print(np.sum(np.count_nonzero(sample_label == 3., axis=-1)) / (288 * 288 * 30))
        print(np.sum(np.count_nonzero(sample_label == 4., axis=-1)) / (288 * 288 * 30))
        print()

        sample = []

        for i in range(len(sample_data)):

            for _ in range(10):

                start_i = random.randint(0, sample_data.shape[1] - 100)
                start_j = random.randint(0, sample_data.shape[2] - 100)

                patch_data = sample_data[i, start_i:start_i+100, start_j:start_j+100, :]
                patch_label = sample_label[i, start_i:start_i+100, start_j:start_j+100]

                sample.append({
                    'X' : patch_data,
                    'Y' : patch_label
                })

        tfrm.save_record(f'{path}/{data_purpose}/{folder}', sample)


def test():

    tfrm = TFRecordsManager()

    dataset = tfrm.load_datasets_without_batching(TFRECORD_PATH)['train']
    dataset = dataset.padded_batch(20, padded_shapes={'X' : (300, 300, 2), 'Y' : (300, 300)})

    for i in dataset:
        print(i['X'].shape)
        print(i['Y'].shape)



if __name__ == '__main__':
    create_tfrecords()
    #main_patches()
    #main()
    #test()


'''

    print(os.listdir(DATA_FOLDER))

    # Get each of the training and validation data images and labels
    example = [
        {
            'X' : [1, 6],
            'Y' : [2]
        },
        {
            'X': [2, 8],
            'Y': [4]
        }
    ]

    tfrm.save_record(path + 'train/example', example)

'''