from mbmvpa.core.engine import run_mbmvpa
from pathlib import Path

#bids_layout = "tutorial_data/ccsl_prl"
bids_layout = "/data2/project_modelbasedMVPA/PRL"
report_path = "tutorial_report"
Path(report_path).mkdir(exist_ok=True)

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.dd_hyperbolic requirements ##
    if row["outcome"] == 0:
        row["outcome"] = -1
    row["onset"] = row["time_feedback"]
    row["duration"] = row["time_wait"] - row["time_feedback"]
    
    if row['choice'] in ['1','2']:
        row['choice'] = int(row['choice'])
        row["outcome"] = int(row["outcome"])
        
    return row

def example_filter(row):
    return row['choice'] in [1,2]


_ = run_mbmvpa(bids_layout=bids_layout,
               mvpa_model='elasticnet',
               dm_model='prl_fictitious_rp_woa',
               feature_name='zoom2',
               task_name='prl',
               process_name='PEchosen',
               report_path=report_path,
               n_core=4,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices=[2],
              adjust_function= example_adjust,
              filter_function=example_filter,
              overwrite=False)