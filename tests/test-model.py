from mbmvpa.data.loader import BIDSDataLoader
from mbmvpa.models.mvpa_mlp import MLP
from pathlib import Path
import pdb

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)
X,y = loader.get_total_data()
voxel_mask = loader.get_voxel_mask()

MVPA_model = MLP(X=X,
                y=y,
                n_batch=2,
                n_repeat=2,
                n_sample=10000,
                n_epoch=20,
                n_patience=5,)

coeffs = MVPA_model.run()

print("TEST PASS!")
