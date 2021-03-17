from mbmvpa.preprocessing.preprocess import DataPreprocessor
from time import perf_counter
#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
#save_path = "/data2/project_modelbasedMVPA/PRL"
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


def example_latent(row, param_dict):
    # calculate subjective utility for choosing later option over sooner option
    # hyperbolic discount function is adopted
    
    return row

s = perf_counter()


preprocessor = DataPreprocessor(bids_layout=root,
                               #save_path=save_path,
                               task_name='prl',
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
                               zoom=(2,2,2))

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.y_generator.set_computational_model(ncore=16)