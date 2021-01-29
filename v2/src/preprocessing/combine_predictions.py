import os
import nibabel as nib
import numpy as np


from common import misc


VIEWS = ['axial', 'sagittal', 'coronal']

def combine(data_folder):

    # Check the model and data folders
    data_path =  f'{misc.get_data_path()}/{data_folder}'
    assert os.path.exists(data_path)

    # Check all the predictions are present
    for view in VIEWS:
        view_path = f'{data_path}/{view}'
        assert all(os.path.exists(f'{view_path}/{f}/{f}-pred.nii') for f in os.listdir(view_path))

    combined_path = misc.mkdir(f'{data_path}/combined/')

    for folder in reversed(os.listdir(f'{data_path}/axial/')):

        data = {view: nib.load(f'{data_path}/{view}/{folder}/{folder}-pred.nii').get_fdata() for view in VIEWS}
        label = nib.load(f'{data_path}/coronal/{folder}/{folder}-label.nii')

        data['axial'] = np.moveaxis(data['sagittal'], 0, 2)
        data['sagittal'] = np.moveaxis(data['sagittal'], 0, 1)

        combined_volume = np.concatenate(list(data.values()), axis=-1)

        misc.mkdir(f'{combined_path}/{folder}/')
        misc.save_nii(combined_volume, f'{combined_path}/{folder}/{folder}-3pred.nii')
        misc.save_nii(label, f'{combined_path}/{folder}/{folder}-label.nii')



if __name__ == '__main__':
    combine('3z-normal')