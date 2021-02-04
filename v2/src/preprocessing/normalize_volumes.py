import nibabel as nib
import numpy as np
import os


def main(path):

    # Create records
    for idx, folder in enumerate(os.listdir(path)):
        print(f'Creating normalized volume for folder: [{folder}]')

        data_path = f'{path}/{folder}/Combined.nii'

        # Load data and labels
        data = nib.load(data_path).get_fdata()
        mean, std = np.mean(data, axis=(0, 1, 2)), np.std(data, axis=(0, 1, 2))
        print(mean, std)
        data = (data - mean) / (std * 3)

        nib.Nifti1Image(data, affine=np.eye(4)).to_filename(f'{path}/{folder}/Normalized.nii')



if __name__ == '__main__':
    main('/home/fayd/Fayd/Projects/organ-segmentation/Unlabelled')
    #main('/home/fayd/Fayd/Projects/organ-segmentation/CHAOS')