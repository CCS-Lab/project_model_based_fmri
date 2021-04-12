from time import perf_counter
from mbmvpa.core.engine import run_mbmvpa
from pathlib import Path

import pdb

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"
report_path = "ccsl_prl"
task = 'prl'
feature = "zoom2"
process = 'PEchosen'

Path(report_path).mkdir(exist_ok=True)

s = perf_counter()

dm_model = "prl_fictitious_rp_woa"

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.dd_hyperbolic requirements ##
    if row["outcome"] == 0:
        row["outcome"] = -1
    row["onset"] = row["time_feedback"]
    row["duration"] = row["time_wait"] - row["time_feedback"]
    return row

def example_filter(row):
    return row['choice'] in [1,2]



print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

_ = run_mbmvpa(root=root,
               dm_model=dm_model,
               mask_path=mask_path,
               report_path=report_path,
               task=task,
               feature=feature,
               process=process,
               adjust_function=example_adjust,
               filter_function=example_filter,)

print(f"INFO: elapsed time for data preprocessing: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.summary()


s = perf_counter()
