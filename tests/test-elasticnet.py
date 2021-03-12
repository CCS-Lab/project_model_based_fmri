from mbmvpa.data.loader import BIDSDataLoader
from mbmvpa.models.mvpa_elasticnet import elasticnet,elasticnet_crossvalidation
from pathlib import Path
import pdb

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)
X,y = loader.get_data(subject_wise=False)
voxel_mask = loader.get_voxel_mask()

coef = elasticnet(X=X,
                  y=y,
                  voxel_mask=voxel_mask,
                  save_path='tests',
                  n_repeat=2)

X_dict,y_dict = loader.get_data(subject_wise=True)

metrics_train, metrics_test, coefs_train = elasticnet_crossvalidation(X_dict,
                                                                     y_dict,
                                                                     voxel_mask=voxel_mask,
                                                                     method='5-fold',
                                                                     n_cv_repeat=2,
                                                                     n_repeat=10,
                                                                     cv_save_path='tests')


print("TEST PASS!")
