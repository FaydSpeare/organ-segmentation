import nibabel as nib

VOLUME_PATH = '/home/fayd/Data/Unlabelled/data/sub-10010_t1_vibe_dixon_tra_bh_3_groups_2_20160607180944_14_Eq_1.nii'

if __name__ == '__main__':

    data = nib.load(VOLUME_PATH).get_fdata()
    print()