import os
import nibabel as nib
import numpy as np
import math
import random

from tf_utils.tfrecords import TFRecordsManager
from tf_utils import misc

DATA_FOLDER = '/home/fayd/Data/CHAOS'
TFRECORD_PATH = '/home/fayd/Fayd/Projects/organ-segmentation/tfrecords/'

def main():

    tfrm = TFRecordsManager()

    misc.save_json(TFRECORD_PATH + 'params.json', {
        'data_purposes': ['train', 'val'],
        'data_keys': {
            'X': 'float32',
            'Y': 'float32'
        }
    })

    for data_purpose in ['train', 'val']:
        if not os.path.isdir(TFRECORD_PATH + data_purpose):
            os.mkdir(TFRECORD_PATH + data_purpose)

    total_samples = len(os.listdir(DATA_FOLDER))
    split = math.floor(0.8 * total_samples)

    for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
        print(f'Creating TFRecord for folder: [{folder}]')

        data_purpose = 'train' if idx <= split else 'val'

        data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'
        label_path = f'{DATA_FOLDER}/{folder}/ground.nii'

        sample_data = nib.load(data_path).get_fdata().astype(np.float32)
        sample_label = nib.load(label_path).get_fdata().astype(np.float32) / 63.0

        # Rearrange data
        sample_data = np.moveaxis(sample_data, 2, 0)
        sample_label = np.moveaxis(sample_label, 2, 0)

        sample = [{'X': sample_data[i], 'Y': sample_label[i]} for i in range(len(sample_data))]
        tfrm.save_record(f'{TFRECORD_PATH}/{data_purpose}/{folder}', sample)


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

        sample_data = nib.load(data_path).get_fdata().astype(np.float32)
        sample_label = nib.load(label_path).get_fdata().astype(np.float32) / 63.0

        # Rearrange data
        sample_data = np.moveaxis(sample_data, 2, 0)
        sample_label = np.moveaxis(sample_label, 2, 0)

        sample = []

        for i in range(len(sample_data)):

            for _ in range(10):

                start_i = random.randint(0, sample_data.shape[1] - 100)
                start_j = random.randint(0, sample_data.shape[2] - 100)

                patch_data = sample_data[:, start_i:start_i+100, start_j:start_j+100, :]
                patch_label = sample_label[:, start_i:start_i+100, start_j:start_j+100]

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
    main_patches()
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