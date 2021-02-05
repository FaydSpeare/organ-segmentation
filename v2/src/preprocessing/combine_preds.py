import os
import nibabel as nib
import numpy as np


from common import misc


VIEWS = ['axial', 'sagittal', 'coronal']


def combine(data_folder, prefixes):

    # Check the model and data folders
    data_path =  f'{misc.get_data_path()}/{data_folder}'
    assert os.path.exists(data_path)

    # Check all the predictions are present
    for view in VIEWS:
        view_path = f'{data_path}/{view}'
        if not all(os.path.exists(f'{view_path}/{f}/{f}-{prefixes[view]}-pred.nii') for f in os.listdir(view_path)):
            print(f'predictions missing for some samples in the {view} view.')
            exit(3)

    combined_path = misc.mkdir(f'{data_path}/combined/')

    for folder in reversed(os.listdir(f'{data_path}/axial/')):

        data = {view: nib.load(f'{data_path}/{view}/{folder}/{folder}-{prefixes[view]}-pred.nii').get_fdata() for view in VIEWS}
        label = nib.load(f'{data_path}/coronal/{folder}/{folder}-label.nii')

        data['axial'] = np.moveaxis(data['axial'], 0, 2)
        data['sagittal'] = np.moveaxis(data['sagittal'], 0, 1)

        combined_volume = np.stack([data['axial'], data['sagittal'], data['coronal']], axis=-1)

        misc.mkdir(f'{combined_path}/{folder}/')
        misc.save_nii(combined_volume, f'{combined_path}/{folder}/{folder}-3pred.nii')
        misc.save_nii(label, f'{combined_path}/{folder}/{folder}-label.nii')



if __name__ == '__main__':
    view_prefixes = {
        'axial'    : 'BDICE',
        'sagittal' : 'BDICE',
        'coronal'  : 'BDICE'
    }
    combine('3x_normal', view_prefixes)