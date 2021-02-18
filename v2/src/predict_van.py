import os
import numpy as np
import nibabel as nib
import math
import tensorflow as tf

from model.network import ViewAggregation
from common import misc


VIEWS = ['axial', 'sagittal', 'coronal']


def predict(model_folder, data_folder, prefixes):

    # Check the model and data folders
    data_path =  f'{misc.get_data_path()}/{data_folder}'
    model_path = f'{misc.get_checkpoint_path()}/{model_folder}'

    assert os.path.exists(data_path)
    assert os.path.exists(model_path)

    # Load model
    # TODO load in from params
    model = ViewAggregation(num_classes=5)
    model.load_weights(f'{model_path}/model_weights').expect_partial()

    for idx, folder in enumerate(os.listdir(f'{data_path}/axial')):
        print(f'Predicting for folder: [{folder}]', flush=True)
        if folder != '10':
            continue

        data = {view: nib.load(f'{data_path}/{view}/{folder}/{folder}-{prefixes[view]}-pred.nii').get_fdata() for view in VIEWS}
        data['axial'] = np.moveaxis(data['axial'], 0, 2)
        data['sagittal'] = np.moveaxis(data['sagittal'], 0, 1)

        A, C, S = 80, 288, 288

        full_prediction = np.zeros((288, 288, 80))
        A_PATCH, C_PATCH, S_PATCH = 80, 72, 72

        for a in range(A // A_PATCH):
            for c in range(C // C_PATCH):
                for s in range(S // S_PATCH):

                    print(a, c, s)


                    a_start = a * A_PATCH
                    c_start = c * C_PATCH
                    s_start = s * S_PATCH

                    a_finish = a_start + A_PATCH
                    c_finish = c_start + C_PATCH
                    s_finish = s_start + S_PATCH

                    patch_data = {view: data[view][c_start:c_finish, s_start:s_finish, a_start:a_finish, :] for view in VIEWS}
                    patch_data = np.concatenate([patch_data['axial'], patch_data['sagittal'], patch_data['coronal']],
                                                axis=-1).astype(np.float32)

                    patch_prediction = model.predict(tf.expand_dims(patch_data, axis=0))

                    patch_prediction = np.eye(patch_prediction.shape[-1])[patch_prediction.argmax(axis=-1)]
                    num_classes = patch_prediction.shape[-1]
                    patch_prediction *= [i * math.floor(255. / num_classes) for i in range(num_classes)]
                    patch_prediction = np.sum(patch_prediction, axis=-1).astype(np.int8)

                    full_prediction[c_start:c_finish, s_start:s_finish, a_start:a_finish] = patch_prediction
                    print()

        print('Saving prediction')
        full_prediction = np.moveaxis(full_prediction, 2, 0)
        misc.save_nii(full_prediction, f'{data_path}/{folder}-van-seg.nii')
        #exit(5)





if __name__ == '__main__':
    predict('(B-VAN_3)-(3x_normal_combined)-(Feb-10-135847)', '3x_normal', {
        'axial': 'BDICE_2',
        'sagittal': 'BDICE_2',
        'coronal': 'BDICE_2'
    })