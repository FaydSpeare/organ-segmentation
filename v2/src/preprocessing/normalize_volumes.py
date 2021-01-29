import nibabel as nib
import numpy as np
import os

DATA_FOLDER = '/home/fayd/Fayd/Projects/organ-segmentation/CHAOS'

def main():

    # Create records
    for idx, folder in enumerate(os.listdir(DATA_FOLDER)):
        print(f'Creating normalized volume for folder: [{folder}]')

        data_path = f'{DATA_FOLDER}/{folder}/Combined.nii'

        # Load data and labels
        data = nib.load(data_path).get_fdata()
        mean, std = np.mean(data, axis=(0, 1, 2)), np.std(data, axis=(0, 1, 2))
        print(mean, std)
        data = (data - mean) / (std * 3)

        nib.Nifti1Image(data, affine=np.eye(4)).to_filename(f'{DATA_FOLDER}/{folder}/Normalized.nii')



if __name__ == '__main__':
    main()