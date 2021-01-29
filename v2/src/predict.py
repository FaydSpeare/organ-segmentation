import os
import numpy as np
import nibabel as nib
import math

from model.network import CDFNet
from common import misc

def predict(model_folder, data_folder):

    # Check the model and data folders
    data_path =  f'{misc.get_data_path()}/{data_folder}'
    model_path = f'{misc.get_checkpoint_path()}/{model_folder}'

    assert os.path.exists(data_path)
    assert os.path.exists(model_path)

    # Load model
    # TODO load in from params
    model = CDFNet(num_filters=64, num_classes=5)
    model.load_weights(f'{model_path}/model_weights').expect_partial()

    for folder in reversed(os.listdir(data_path)):

        print(f"Saving segmentation for folder: [{folder}]")

        sample = nib.load(f'{data_path}/{folder}/{folder}-data.nii').get_fdata()

        prediction = []
        for batch in np.array_split(sample, 10):
            prediction.append(model.predict(batch))

        prediction = np.concatenate(prediction)
        misc.save_nii(prediction, f'{data_path}/{folder}/{folder}-pred.nii')

        # Collapse the probabilities into a one hot encoding then multiply and sum
        prediction = np.eye(prediction.shape[-1])[prediction.argmax(axis=-1)]
        num_classes = prediction.shape[-1]
        prediction *= [i * math.floor(255. / num_classes) for i in range(num_classes)]
        prediction = np.sum(prediction, axis=-1)

        misc.save_nii(prediction, f'{data_path}/{folder}/{folder}-seg.nii')



if __name__ == '__main__':
    predict("GDICE-normalized_axial-ckp_20210129_155858", '3z-normal/axial')