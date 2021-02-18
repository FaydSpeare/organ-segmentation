import os
from intensity_normalization.normalize import nyul
from intensity_normalization.utilities import io


from common import misc

print('Start')

chaos_path = misc.get_chaos_path()


for name in ['InPhase', 'OutPhase']:

    print(f'Working on: {name}')
    input_files, output_files = list(), list()
    for folder in os.listdir(chaos_path):
        input_files.append(f'{chaos_path}/{folder}/{name}.nii')
        output_files.append(f'{chaos_path}/{folder}/{name}_HM.nii')

    print('Training')
    mask_files = [None] * len(input_files)
    standard_scale, percs = nyul.train(input_files, mask_files)

    print('Normalizing')
    for img_fn, out_fn in zip(input_files, output_files):
        print(img_fn, '->', out_fn)
        _, base, _ = io.split_filename(img_fn)
        img = io.open_nii(img_fn)
        normalized = nyul.do_hist_norm(img, percs, standard_scale, mask=None)
        io.save_nii(normalized, out_fn, is_nii=True)

print('Done')