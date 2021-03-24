from time import perf_counter
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.elasticnet import MVPA_ElasticNet
from mbmvpa.models.report import build_elasticnet_report_functions
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path

#root = load_example_data("tom")
root = "/data2/project_model_based_fmri/piva_dd"
save_path = "/data2/project_modelbasedMVPA/piva_dd_mbmvpa"
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"
report_path = "piva_dd"

Path(report_path).mkdir(exist_ok=True)

loader = BIDSDataLoader(layout=save_path)
X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()
print(voxel_mask.shape)

#loader = BIDSDataLoader(layout=root)
X_dict,y_dict = loader.get_data(subject_wise=True)

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
                    cv_save_path=report_path,
                    task_name="dd",
                    report_function_dict=report_function_dict)


s = perf_counter()
model_cv.run()
print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

'''
model_cv = MVPA_CV(X_dict,
                    y_dict,
                    model,
                    model_param_dict={},
                    method='loso',
                    n_cv_repeat=2,
                    cv_save=True,
                    cv_save_path=report_path,
                    task_name="ddt",
                    report_function_dict=report_function_dict)

model_cv.run()
'''