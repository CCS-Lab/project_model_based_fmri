from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path
import pdb

## Test fmri processing from root path

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)
X,y = loader.get_total_data()
print("X", X.shape)
print("y", y.shape)
print("TEST PASS!")