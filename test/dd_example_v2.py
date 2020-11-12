from model_based_mvpa.models.regressor import *
from model_based_mvpa.utils.coef2map import *
import time

prep_path = Path('/data2/project_modelbasedMVPA/temp_2')
data_path_list = [prep_path / f'X_{i:02d}.npy' for i in range(1,21)]

X = np.concatenate([np.load(data_path) for data_path in data_path_list],0)

y = np.load('/data2/project_model_based_fmri/y_dd.npy',allow_pickle=True)

y = np.concatenate(y,0)

X = X.reshape(-1,X.shape[-1])
y = y.flatten()

masked_data = nib.load(prep_path / 'masked_data.nii.gz')

print(time.strftime('%c', time.localtime(time.time())))

coefs = mlp_regression(X, y,
                       layer_dims=[1024, 1024],
                       activation_func='linear',
                       dropout_rate=0.5,
                       epochs=100,
                       patience=10,
                       batch_size=64,
                       N=10,
                       verbose=1)

task_name = 'piva2019_mlp_10_v2'
result = get_map(coefs, masked_data, task_name, map_type='z', save_path='.', smoothing_sigma=1)
print(time.strftime('%c', time.localtime(time.time())))

coefs = elasticnet(X, y, 
             alpha=0.001,
             n_jobs=16,
             N=1,
             verbose=0)

task_name = 'piva2019_elasticnet_v2'
result = get_map(coefs, masked_data, task_name, map_type='z', save_path='.', smoothing_sigma=1)
print(time.strftime('%c', time.localtime(time.time())))