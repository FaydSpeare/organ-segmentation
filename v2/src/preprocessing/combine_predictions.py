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
        #assert all(os.path.exists(f'{view_path}/{f}/{f}-pred.nii') for f in os.listdir(view_path))

    combined_path = misc.mkdir(f'{data_path}/combined/')

    for folder in reversed(os.listdir(f'{data_path}/axial/')):

        if folder != '37': continue

        data = {view: nib.load(f'{data_path}/{view}/{folder}/{folder}--pred.nii').get_fdata() for view in VIEWS}
        label = nib.load(f'{data_path}/coronal/{folder}/{folder}-label.nii').get_fdata()

        data['axial'] = np.moveaxis(data['axial'], 0, 2)
        data['sagittal'] = np.moveaxis(data['sagittal'], 0, 1)
        data['coronal'] = data['coronal']
        combined_volume = np.zeros(data['axial'].shape[:-1])
        #exit(2)
        for i in range(288):
            print(i)
            for j in range(288):
                for k in range(144):
                    #stacked = np.stack([data['axial'][i][j][k], data['sagittal'][i][j][k], data['coronal'][i][j][k]], axis=-1)
                    #combined_volume[i][j][k] = np.argmax(np.max(stacked, axis=-1)) * 63.0
                    combined_volume[i][j][k] = np.argmax(0.5*data['axial'][i][j][k] + 0.25*data['sagittal'][i][j][k] + 0.25*data['coronal'][i][j][k]) * 63.0


        #combined_volume = np.stack([data['axial'], data['sagittal'], data['coronal']], axis=-1)

        misc.mkdir(f'{combined_path}/{folder}/')
        misc.save_nii(combined_volume, f'{combined_path}/{folder}/{folder}-3pred.nii')
        misc.save_nii(label, f'{combined_path}/{folder}/{folder}-label.nii')



if __name__ == '__main__':
    combine('view-agg-data')