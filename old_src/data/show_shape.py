import os
import nibabel as nib

DATA_FOLDER = '/home/fayd/Data/CHAOS'

if __name__ == '__main__':

    for folder in os.listdir(DATA_FOLDER):
        print(folder, nib.load(DATA_FOLDER + '/' + folder + '/InPhase.nii').get_fdata().shape)
        #print(nib.load(DATA_FOLDER + '/' + folder + '/OutPhase.nii').get_fdata().shape)