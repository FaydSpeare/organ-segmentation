import dicom2nifti
import os
from skimage.io import imread_collection, concatenate_images
import nibabel as nib
import numpy as np

DATA_FOLDER = "/home/fayd/Data/CHAOS_Train_Sets/Train_Sets/MR"
OUTPUT_FOLDER = '/home/fayd/Data/CHAOS'
TYPE = 'T1DUAL'

def convert_ct():

    # Create output folder
    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    for folder_name in os.listdir(DATA_FOLDER):

        output_folder = f'{OUTPUT_FOLDER}/{folder_name}'
        input_folder = f'{DATA_FOLDER}/{folder_name}/{TYPE}'

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        for phase in ['InPhase', 'OutPhase']:
            dicom_dir = input_folder + '/DICOM_anon/' + phase
            dicom2nifti.convert_directory(dicom_dir, output_folder, compression=False)

            for file in os.listdir(output_folder):
                if file.endswith(".nii") and not file.startswith("InPhase"):
                    os.rename(output_folder + '/' + file ,output_folder + '/' + phase + '.nii')

        data = list()
        for phase in ['InPhase', 'OutPhase']:
            data.append(nib.load(output_folder + '/' + phase + '.nii').get_fdata().astype(np.float32))

        # Combine the phases as channels
        data = np.stack(data, axis=-1)
        nii_data = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(nii_data, output_folder + '/Combined.nii')

        # Combine collection of 2D images
        images = input_folder + '/Ground/*.png'
        label = concatenate_images(imread_collection(images))
        label = np.moveaxis(label, 0, -1) # put depth last
        label = np.rot90(label, k=3) # 270 degree rotation

        # Save image 3D array as nii
        nii_label = nib.Nifti1Image(label, affine=np.eye(4))
        nii_label.to_filename(output_folder + '/ground.nii')


if __name__ == '__main__':
    convert_ct()