from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path
import pdb

## Test fmri processing from root path

root = Path('tests/test_example')
process_name = "testprocess"
feature_name = "testfeature"
loader = BIDSDataLoader(layout=root,
                        process_name=process_name,
                        feature_name=feature_name,)
X,y = loader.get_data(subject_wise=False)
voxel_mask = loader.get_voxel_mask()
print("X", X.shape)
print("y", y.shape)
print("TEST PASS!")
X_dict,y_dict = loader.get_data(subject_wise=True)

for key,data in X_dict.items():
    print(f"X-{key}", data.shape)
    print(f"y-{key}", y_dict[key].shape)