import os
import nibabel as nib
import numpy as np

from tf_utils.tfrecords import TFRecordsManager
from tf_utils import misc

DATA_FOLDER = '/home/fayd/Data/CHAOS'
TFRECORD_PATH = '/home/fayd/Fayd/Projects/organ-segmentation/tfrecords/'

def main():

    tfrm = TFRecordsManager()

    misc.save_json(TFRECORD_PATH + 'params.json', {
        'data_purposes': ['train'],
        'data_keys': {
            'X': 'float32',
            'Y': 'float32'
        }
    })

    if not os.path.isdir(TFRECORD_PATH + 'train'):
        os.mkdir(TFRECORD_PATH + 'train')

    for folder in os.listdir(DATA_FOLDER):
        print(f'Creating TFRecord for folder: [{folder}]')

        data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'
        label_path = f'{DATA_FOLDER}/{folder}/ground.nii'

        sample_data = nib.load(data_path).get_fdata().astype(np.float32)
        sample_label = nib.load(label_path).get_fdata().astype(np.float32)

        # Rearrange data
        sample_data = np.moveaxis(sample_data, 2, 0)
        sample_label = np.moveaxis(sample_label, 2, 0)

        sample = [{'X': sample_data[i], 'Y': sample_label[i]} for i in range(len(sample_data))]
        tfrm.save_record(f'{TFRECORD_PATH}/train/{folder}', sample)


def test():

    tfrm = TFRecordsManager()

    dataset = tfrm.load_datasets_without_batching(TFRECORD_PATH)['train']
    dataset = dataset.padded_batch(20, padded_shapes={'X' : (300, 300, 2), 'Y' : (300, 300)})

    for i in dataset:
        print(i['X'].shape)
        print(i['Y'].shape)



if __name__ == '__main__':
    main()
    test()


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