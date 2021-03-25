from time import perf_counter
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from mbmvpa.models.mvpa_elasticnet import elasticnet_crossvalidation
from mbmvpa.data.loader import BIDSDataLoader
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.elasticnet import MVPA_ElasticNet
from mbmvpa.utils.report import build_elasticnet_report_functions
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path
#root = load_example_data("tom")

root = "/data2/project_modelbasedMVPA/ds000005"
save_path = "/data2/project_modelbasedMVPA/tom"
Path(save_path).mkdir(exist_ok=True)
s = perf_counter()
dm_model = 'ra_prospect'

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row

def example_filter(row):
    # include all trial data
    return True

def example_latent(row, param_dict):
    ## calculate subjectives utility for choosing Gamble over Safe option
    ## prospect theory with loss aversion and risk aversion is adopted
    modulation = (row["gain"] ** param_dict["rho"]) - (param_dict["lambda"] * (row["loss"] ** param_dict["rho"]))
    row["modulation"] = modulation
    return row

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")


preprocessor = DataPreprocessor(bids_layout=root,
                               save_path=save_path,
                               adjust_function=example_adjust,
                               filter_function=example_filter,
                               latent_function=example_latent,
                               dm_model=dm_model,
                               mask_threshold=2.58,
                               standardize=True,
                               confounds=[],
                               high_pass=1/128,
                               detrend=False,
                               smoothing_fwhm=6, 
                               zoom=(2,2,2),
                               n_core=24)



s = perf_counter()

preprocessor.preprocess(overwrite=False,n_core=24)
#preprocessor.preprocess(overwrite=False,n_core=16)
#preprocessor.X_generator.run(overwrite=True)
print(f"INFO: elapsed time for data preprocessing: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.summary()


s = perf_counter()

loader = BIDSDataLoader(layout=root)


print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")


report_path = "tom_prospect"

Path(report_path).mkdir(exist_ok=True)

loader = BIDSDataLoader(layout=save_path)
X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()
#loader = BIDSDataLoader(layout=root)

model = MVPA_ElasticNet(alpha=0.001,
                         n_samples=50000,
                         shuffle=True,
                         max_lambda=50,
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
                    n_cv_repeat=1,
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