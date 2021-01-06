from mbmvpa.preprocessing.bids import bids_preprocess
from pathlib import Path
import pdb
root = Path('tests/test_example')
X, voxel_mask, layout, data_root = bids_preprocess(root, smoothing_fwhm=None, zoom=(2, 2, 2), ncore=2, nthread=4)
print("TEST PASS!")