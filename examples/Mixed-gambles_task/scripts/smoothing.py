## -*- coding: utf-8 -*- 
import nibabel as nib

from bids import BIDSLayout
from tqdm import tqdm

from nilearn.image import smooth_img
import argparse

import time

def main():
    parser = argparse.ArgumentParser(description='Usage: python post_prep.py --data_dir=YOUR_DIRECTORY, --fwhm=FWHM')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='the directory path including data after fMRIPrep')
    parser.add_argument('--fwhm', required=True, type=float,
                        help='fwhm')
    args = parser.parse_args()
    print(args)

    layout = BIDSLayout(args.data_dir, derivatives=True)

    img_files = layout.derivatives['fMRIPrep'].get(
        return_type='file', suffix='bold', extension='nii.gz')

    s0 = time.time()
    print('smoothing start..')
    smoothed_imgs = smooth_img(img_files, fwhm=args.fwhm)
    
    c = 0
    for smoothed_img, img_file in zip(smoothed_imgs, img_files):
        nib.save(smoothed_img, img_file.replace('.nii.gz', '_smoothed.nii.gz'))
        c += 1

    assert len(img_files) == c, 'error..'

    print(f'{c} images were smoothed')
    print(f'smoothing finished..! elapsed time: {time.time() - s0:.2f} seconds')

if __name__ == '__main__':
    main()
