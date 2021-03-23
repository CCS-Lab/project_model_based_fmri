from time import perf_counter
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from mbmvpa.models.mvpa_elasticnet import elasticnet_crossvalidation
from pathlib import Path

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"
report_path = "ccsl_prl"

Path(report_path).mkdir(exist_ok=True)

s = perf_counter()

dm_model = "prl_fictitious_rp_woa"

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.dd_hyperbolic requirements ##
    if row["outcome"] == 0:
        row["outcome"] = -1
    row["onset"] = row["time_onset"]
    row["duration"] = row["time_choice"] - row["time_onset"]
    return row

def example_filter(row):
    # in the paper, the condition for trial varies in a single run,
    # agent == 0 for making a choice for him or herself
    # agent == 1 for making a choice for other
    # to consider only non-social choice behavior, select only the cases with agent == 0
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
        modulations.append(ev[0])
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        try:
            if PE >= 0:
                ev[choice-1] += eta_pos * PE;
                ev[2 - choice] += eta_pos * PEnc;
            else :
                ev[choice-1] += eta_neg * PE;
                ev[2 - choice] += eta_neg * PEnc;
        except:
            print(PE)
            
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]



print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

s = perf_counter()

preprocessor = DataPreprocessor(bids_layout=root,
                               #save_path=save_path,
                               mask_path=mask_path,
                               task_name='prl',
                               adjust_function=example_adjust,
                               filter_function=example_filter,
                               modulation_dfwise=example_modulation_dfwise,
                               dm_model=dm_model,
                               mask_threshold=2.58,
                               process_name="q_value",
                               standardize=True,
                               confounds=[],
                               high_pass=1/128,
                               detrend=False,
                               smoothing_fwhm=6, 
                               zoom=(1,1,1),
                               n_core= 24)

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")


s = perf_counter()

preprocessor.preprocess(overwrite=True,n_core=24)
#preprocessor.preprocess(overwrite=False,n_core=16)
#preprocessor.X_generator.run(overwrite=True)
print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.summary()

from mbmvpa.data.loader import BIDSDataLoader

s = perf_counter()

loader = BIDSDataLoader(layout=save_path,
                       process_name="q_value")
voxel_mask = loader.get_voxel_mask()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

#loader = BIDSDataLoader(layout=root)
X_dict,y_dict = loader.get_data(subject_wise=True)


_ = elasticnet_crossvalidation(X_dict,
                               y_dict,
                               voxel_mask=voxel_mask,
                               cv_save_path=report_path,
                               method='5-fold',
                               n_cv_repeat=5,
                               n_repeat=5)

_ = elasticnet_crossvalidation(X_dict,
                               y_dict,
                               voxel_mask=voxel_mask,
                               cv_save_path=report_path,
                               method='loso',
                               n_cv_repeat=5,
                               n_repeat=5)

