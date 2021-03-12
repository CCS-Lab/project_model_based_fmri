from time import perf_counter
from mbmvpa.models.mvpa_elasticnet import elasticnet, elasticnet_crossvalidation
from mbmvpa.data.loader import BIDSDataLoader


root = "/data2/project_modelbasedMVPA/ds000005"

s = perf_counter()
loader = BIDSDataLoader(layout=root)
X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()

print(f"X,y,voxel_mask are loaded. elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

s = perf_counter()

metrics_train, metrics_test, coefs_train = elasticnet_crossvalidation(X_dict,
                                                                     y_dict,
                                                                     voxel_mask=voxel_mask,
                                                                     method='5-fold',
                                                                     n_cv_repeat=10,
                                                                     n_repeat=10,
                                                                     cv_save_path='.')
print(f"Done. elapsed time: {(perf_counter()-s) / 60:.2f} minutes")