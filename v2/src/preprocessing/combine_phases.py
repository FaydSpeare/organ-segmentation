import numpy as np
import os
import nibabel as nib
import tensorflow as tf

from common import misc

def combine(path):

    for folder in os.listdir(path):

        in_data = nib.load(f'{path}/{folder}/inPhase.nii').get_fdata()
        out_data = nib.load(f'{path}/{folder}/outPhase.nii').get_fdata()

        combined = np.stack([in_data, out_data], axis=-1)
        combined = np.moveaxis(combined, 2, 0)
        combined = tf.image.resize_with_crop_or_pad(combined, 288, 288)
        misc.save_nii(combined, f'{path}/{folder}/Combined.nii')



if __name__ == '__main__':
    combine('/home/fayd/Fayd/Projects/organ-segmentation/Unlabelled')