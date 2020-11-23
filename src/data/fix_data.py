import nibabel as nib
import numpy as np

DATA_FOLDER = "/home/fayd/Data/CHAOS_Converted_Train_Sets/CT/1/"

def main():
    data = nib.load(DATA_FOLDER + 'preds.nii').get_fdata().astype(np.float32)
    data = np.swapaxes(data, 0, -1)
    data = np.swapaxes(data, 1, 2)
    data[data > 0.5] = 255.
    data[data < 0.5] = 0.


    nii_label = nib.Nifti1Image(data[0], affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/fixed_preds_1.nii')

    nii_label = nib.Nifti1Image(data[1], affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/fixed_preds_2.nii')

if __name__ == '__main__':
    main()