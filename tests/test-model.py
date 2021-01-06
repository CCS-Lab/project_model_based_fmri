from mbmvpa.data.loader import prepare_dataset
from mbmvpa.models.model import *
from pathlib import Path
import pdb

data_root = Path('tests/test_example/derivatives/fmriprep')
X, y, voxel_mask = prepare_dataset(data_root)

MVPA_model = MLP(X=X,
                y=y,
                n_batch=2,
                n_repeat=2,
                n_sample=10000,
                n_epoch=20,
                n_patience=5,)

coeffs = MVPA_model.run()

print("TEST PASS!")
