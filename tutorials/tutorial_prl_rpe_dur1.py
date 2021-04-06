from time import perf_counter
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from mbmvpa.models.mvpa_elasticnet import elasticnet_crossvalidation
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path

import pdb

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"
report_path = "ccsl_prl"
process_name = "rpedur1"

Path(report_path).mkdir(exist_ok=True)

s = perf_counter()

dm_model = "prl_fictitious_rp_woa"

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.dd_hyperbolic requirements ##
    if row["outcome"] == 0:
        row["outcome"] = -1
    row["onset"] = row["time_feedback"]
    row["duration"] = 1 #row["time_wait"] - row["time_feedback"]
    return row

def example_filter(row):
    return row['choice'] in [1,2]


def example_modulation_dfwise(df_events, param_dict):
    
    df_events = df_events.sort_values(by="onset")
    
    ev = [0,0]
    choices = df_events['choice'].to_numpy()
    outcome = df_events['outcome'].to_numpy()
    modulations = []
    #print(param_dict['eta_pos'], param_dict['eta_neg'])
    eta_pos = float(param_dict['eta_pos'])
    eta_neg = float(param_dict['eta_neg'])
    #print(eta_pos, eta_neg)
    for choice, outcome in zip(choices,outcome):
        choice = int(choice)
        outcome = int(outcome)
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        
        modulations.append(PE)
        
        if PE >= 0:
            ev[choice-1] += eta_pos * PE;
            ev[2 - choice] += eta_pos * PEnc;
        else :
            ev[choice-1] += eta_neg * PE;
            ev[2 - choice] += eta_neg * PEnc;
    
    df_events["modulation"] = modulations
    return df_events[['onset','duration','modulation']]

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor = DataPreprocessor(bids_layout=root,
                               #save_path=save_path,
                               mask_path=mask_path,
                               task_name='prl',
                               adjust_function=example_adjust,
                               filter_function=example_filter,
                               modulation_dfwise=example_modulation_dfwise,
                               dm_model=dm_model,
                               mask_threshold=2.58,
                               process_name=process_name,
                               standardize=True,
                               confounds=[],
                               high_pass=1/128,
                               detrend=False,
                               smoothing_fwhm=6, 
                               zoom=(2,2,2),
                               n_core= 24)



s = perf_counter()

#preprocessor.preprocess(overwrite=False,n_core=24)
preprocessor.y_generator.run(overwrite=True,process_name=process_name)
#preprocessor.preprocess(overwrite=False,n_core=16)
#preprocessor.X_generator.run(overwrite=True)
print(f"INFO: elapsed time for data preprocessing: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.summary()


s = perf_counter()

loader = BIDSDataLoader(layout=root,
                       process_name=process_name)


print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

from time import perf_counter
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.elasticnet import MVPA_ElasticNet
from mbmvpa.utils.report import build_elasticnet_report_functions
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
report_path = "ccsl_prl"
task_name = "prl"
process_name = "rpedur1"

Path(report_path).mkdir(exist_ok=True)

'''
subjects = ['01','02','03','04','05','06',
            '07','08','09','10', '11', '12',
            ]
'''
subjects = None


loader = BIDSDataLoader(layout=root, process_name=process_name, subjects=subjects, normalizer='minmax')
X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()

model = MVPA_ElasticNet(alpha=0.001,
                         n_samples=100000,
                         shuffle=True,
                         max_lambda=50,
                         min_lambda_ratio=1e-4,
                         lambda_search_num=100,
                         n_jobs=16,
                         n_splits=5)

report_function_dict = build_elasticnet_report_functions(voxel_mask,
                                                         confidence_interval=.99,
                                                         n_coef_plot=150,
                                                         task_name=task_name+"-"+process_name,
                                                         map_type='z',
                                                         sigma=1
                                                         )

model_cv = MVPA_CV(X_dict,
                    y_dict,
                    model,
                    model_param_dict={},
                    method='5-fold',
                    n_cv_repeat=5,
                    cv_save=True,
                    cv_save_path=report_path,
                    task_name=task_name,
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