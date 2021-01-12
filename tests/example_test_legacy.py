from mbmvpa.data.loader import prepare_dataset
from mbmvpa.models.old_regressor import *
from bids import BIDSLayout
from mbmvpa.utils.coef2map import get_map
import numpy as np

root = "/data2/project_modelbasedMVPA/ds000005"
layout = BIDSLayout(root, derivatives=True)

X, y, voxel_mask = prepare_dataset(layout.derivatives["fMRIPrep"].root)

'''
coefs = mlp_regression(X, y,
                       layout,
                       layer_dims=[1024, 1024],
                       activation="linear",
                       dropout_rate=0.5,
                       epochs=200,
                       patience=20,
                       batch_size=64,
                       N=10,
                       verbose=1)

result = get_map(coefs, voxel_mask, task_name="mlp_legacy_original", map_type="z", save_path=".", sigma=1)
'''
coefs = elasticnet(X, y,
                   layout,
                   n_jobs=16,
                   verbose=1,
                   max_lambda=1,
                   n_samples=5000)

result = get_map(coefs, voxel_mask, task_name="elasticnet_legacy_original", map_type="z", save_path=".", sigma=1)

ids = np.arange(y.flatten().shape[0])
np.random.shuffle(ids)

assert(ids[0] != 0)

y = y[ids]
'''
coefs = mlp_regression(X, y,
                       layout,
                       layer_dims=[1024, 1024],
                       activation="linear",
                       dropout_rate=0.5,
                       epochs=200,
                       patience=20,
                       batch_size=64,
                       N=10,
                       verbose=1)

result = get_map(coefs, voxel_mask, task_name="mlp_legacy_sham", map_type="z", save_path=".", sigma=1)
'''
coefs = elasticnet(X, y,
                   layout,
                   n_jobs=16,
                   verbose=1,
                   max_lambda=1,
                   n_samples=5000)

result = get_map(coefs, voxel_mask, task_name="elasticnet_legacy_sham", map_type="z", save_path=".", sigma=1)
