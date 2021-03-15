import os
from intensity_normalization.normalize import nyul
from intensity_normalization.utilities import io


from common import misc


def match_histograms(reference_path, apply_to_path):

    print('Start')

    for name in ['InPhase', 'OutPhase']:

        print(f'Working on: {name}')
        ref_files = list()
        for folder in os.listdir(reference_path):
            ref_files.append(f'{reference_path}/{folder}/{name}.nii')

        print('Training')
        mask_files = [None] * len(ref_files)
        standard_scale, percs = nyul.train(ref_files, mask_files)

        input_files, output_files = list(), list()
        for folder in os.listdir(apply_to_path):
            input_files.append(f'{apply_to_path}/{folder}/{name}.nii')
            output_files.append(f'{apply_to_path}/{folder}/{name}_HM.nii')

        print('Normalizing')
        for img_fn, out_fn in zip(input_files, output_files):
            print(img_fn, '->', out_fn)
            _, base, _ = io.split_filename(img_fn)
            img = io.open_nii(img_fn)
            normalized = nyul.do_hist_norm(img, percs, standard_scale, mask=None)
            io.save_nii(normalized, out_fn, is_nii=True)

    print('Done')


if __name__ == '__main__':
    match_histograms(misc.get_chaos_path(), '/home/fayd/Fayd/Projects/organ-segmentation/Unlabelled')