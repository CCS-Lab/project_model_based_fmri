from mbmvpa.data.loader import BIDSDataLoader
#from mbmvpa.models.mvpa_elasticnet import elasticnet,elasticnet_crossvalidation
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.tf_mlp import MVPA_MLP
from mbmvpa.utils.report import build_base_report_functions
from pathlib import Path
import pdb

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)

X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()

input_shape = X_dict[list(X_dict.keys())[0]].shape[1]

model = MVPA_MLP(input_shape,
                 layer_dims=[1024, 1024],
                 activation="linear",
                 activation_output="linear",
                 dropout_rate=0.5,
                 val_ratio=0.2,
                 optimizer="adam",
                 loss="mse",
                 learning_rate=0.001,
                 n_epoch = 10,
                 n_patience = 10,
                 n_batch = 4,
                 n_sample = 30000,
                 use_bias = True,
                 use_bipolar_balancing = False)

report_function_dict = build_base_report_functions(voxel_mask,
                             task_name='unnamed',
                             map_type='z',
                             sigma=1
                             )

model_cv = MVPA_CV(X_dict,
                    y_dict,
                    model,
                    model_param_dict={},
                    method='5-fold',
                    n_cv_repeat=2,
                    cv_save=True,
                    cv_save_path="tests",
                    task_name="test",
                    report_function_dict=report_function_dict)

model_cv.run()

model_cv = MVPA_CV(X_dict,
                    y_dict,
                    model,
                    model_param_dict={},
                    method='loso',
                    n_cv_repeat=2,
                    cv_save=True,
                    cv_save_path="tests",
                    task_name="test",
                    report_function_dict=report_function_dict)

model_cv.run()

'''
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

'''
print("TEST PASS!")
