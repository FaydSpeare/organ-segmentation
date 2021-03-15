import numpy as np
import os
import nibabel as nib
import tensorflow as tf

from common import misc

def combine(path):

    for folder in os.listdir(path):
        print(folder)
        in_data = nib.load(f'{path}/{folder}/InPhase.nii')
        in_phase = in_data.get_fdata()
        out_data = nib.load(f'{path}/{folder}/OutPhase.nii')
        out_phase = out_data.get_fdata()

        combined = np.stack([in_phase, out_phase], axis=-1)
        combined = np.moveaxis(combined, 2, 0)
        combined = tf.image.resize_with_crop_or_pad(combined, 288, 288)
        misc.save_nii(combined, f'{path}/{folder}/Combined.nii', header=in_data.header)
        exit(4)



if __name__ == '__main__':
    combine('/home/fayd/Fayd/Projects/organ-segmentation/CHAOS')