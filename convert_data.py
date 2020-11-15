import dicom2nifti
import os
from skimage.io import imread_collection, concatenate_images
import nibabel as nib
import numpy as np

DATA_FOLDER = "/home/fayd/Data/CHAOS_Train_Sets/Train_Sets/"
OUTPUT_FOLDER = '/home/fayd/Data/CHAOS_Converted_Train_Sets/'

def convert_ct():

    input_folder = DATA_FOLDER + 'CT/'
    output_folder = OUTPUT_FOLDER + 'CT/'

    for folder_name in os.listdir(input_folder):
        if not os.path.isdir(output_folder + folder_name):
            os.mkdir(output_folder + folder_name)
        dicom2nifti.convert_directory(input_folder + folder_name, output_folder + folder_name, compression=False)

        # Combine collection of 2D images
        images = input_folder + folder_name + '/Ground/*.png'
        label = concatenate_images(imread_collection(images))
        label = np.flip(label, 0) # reverse order of depth
        label = np.moveaxis(label, 0, -1) # put depth last
        label = np.rot90(label, k=3) # 270 degree rotation

        # Save image 3D array as nii
        nii_label = nib.Nifti1Image(label, affine=np.eye(4))
        nii_label.to_filename(output_folder + folder_name + '/ground.nii')

if __name__ == '__main__':
    convert_ct()