from mbmvpa.data.loader import BIDSDataLoader
#from mbmvpa.models.mvpa_elasticnet import elasticnet,elasticnet_crossvalidation
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.elasticnet import MVPA_ElasticNet
from mbmvpa.models.report import build_elasticnet_report_functions
from pathlib import Path
import pdb

root = Path('tests/test_example')
loader = BIDSDataLoader(layout=root)

X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()

model = MVPA_ElasticNet(alpha=0.001,
                         n_samples=50000,
                         shuffle=True,
                         max_lambda=10,
                         min_lambda_ratio=1e-4,
                         lambda_search_num=100,
                         n_jobs=16,
                         n_splits=5)

report_function_dict = build_elasticnet_report_functions(voxel_mask,
                                                         confidence_interval=.99,
                                                         n_coef_plot=150,
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
