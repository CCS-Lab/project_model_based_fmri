from mbmvpa.data.loader import BIDSDataLoader
from mbmvpa.models.mvpa_elasticnet import elasticnet
from pathlib import Path
import pdb

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)
X,y = loader.get_total_data()
voxel_mask = loader.get_voxel_mask()

coef = elasticnet(X=X,
                  y=y,
                  voxel_mask=voxel_mask,
                  save_path='tests')
            


print("TEST PASS!")
