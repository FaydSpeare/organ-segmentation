import nibabel as nib
import numpy as np

DATA_FOLDER = "/home/fayd/Data/CHAOS/1/"

def main():
    labels = nib.load(DATA_FOLDER + 'preds.nii').get_fdata().astype(np.float32)
    labels = np.swapaxes(labels, 0, 2)
    labels = np.swapaxes(labels, 0, 1)

    # Collapse the probabilities into a one hot encoding
    labels = np.eye(labels.shape[-1])[labels.argmax(axis=-1)]

    # Multiply out and sum
    labels *= [0, 63, 126, 179, 242, 0]
    labels = np.sum(labels, axis=-1)

    nii_label = nib.Nifti1Image(labels, affine=np.eye(4))
    nii_label.to_filename(DATA_FOLDER + '/fixed_preds.nii')

if __name__ == '__main__':
    main()