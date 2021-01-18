from mbmvpa.preprocessing.bold import VoxelFeatureGenerator
from pathlib import Path
import pdb


## Test fmri processing from root path

root = Path('tests/test_example')
X_generator = VoxelFeatureGenerator(bids_layout=root)
X_generator.summary()
X_generator.preprocess(overwrite=True)
print("TEST PASS!")