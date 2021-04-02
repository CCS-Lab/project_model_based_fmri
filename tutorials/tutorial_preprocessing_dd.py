from time import perf_counter
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from mbmvpa.models.mvpa_elasticnet import elasticnet_crossvalidation
from pathlib import Path

#root = load_example_data("tom")
root = "/data2/project_model_based_fmri/piva_dd"
save_path = "/data2/project_modelbasedMVPA/piva_dd_mbmvpa_zoomed_2"
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"

s = perf_counter()

dm_model = "dd_hyperbolic"

def example_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.dd_hyperbolic requirements ##
    if row["delay_left"] >= row["delay_right"]:
        row["delay_later"] = row["delay_left"]
        row["delay_sooner"] = row["delay_right"]
        row["amount_later"] = row["money_left"]
        row["amount_sooner"] = row["money_right"]
        row["choice"] = 1 if row["choice"] == 1 else 0
    else:
        row["delay_later"] = row["delay_right"]
        row["delay_sooner"] = row["delay_left"]
        row["amount_later"] = row["money_right"]
        row["amount_sooner"] = row["money_left"]
        row["choice"] = 1 if row["choice"] == 2 else 0
    return row

def example_filter(row):
    # in the paper, the condition for trial varies in a single run,
    # agent == 0 for making a choice for him or herself
    # agent == 1 for making a choice for other
    # to consider only non-social choice behavior, select only the cases with agent == 0
    return row["agent"] == 0


def example_latent(row, param_dict):
    # calculate subjective utility for choosing later option over sooner option
    # hyperbolic discount function is adopted
    ev_later = row["amount_later"] / (1 + param_dict["k"] * row["delay_later"])
    ev_sooner  = row["amount_sooner"] / (1 + param_dict["k"] * row["delay_sooner"])
    modulation = ev_later - ev_sooner
    row["modulation"] = modulation
    return row



preprocessor = DataPreprocessor(bids_layout=root,
                               save_path=save_path,
                               mask_path=mask_path,
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
                               n_core=4)

print(f"INFO: elapsed time for setting preprocessor: {(perf_counter()-s) / 60:.2f} minutes")



s = perf_counter()

#preprocessor.preprocess(overwrite=True,n_thread=4,n_core=16)
#preprocessor.preprocess(overwrite=False,n_core=16)
#preprocessor.X_generator.run(overwrite=True,n_thread=4)
preprocessor.y_generator.run(overwrite=True)
print(f"INFO: elapsed time for data preprocessing: {(perf_counter()-s) / 60:.2f} minutes")

preprocessor.summary()

from mbmvpa.data.loader import BIDSDataLoader

s = perf_counter()

loader = BIDSDataLoader(layout=save_path)
voxel_mask = loader.get_voxel_mask()
print(voxel_mask.shape)
print(f"INFO: elapsed time for preparing training data: {(perf_counter()-s) / 60:.2f} minutes")

#loader = BIDSDataLoader(layout=root)
X_dict,y_dict = loader.get_data(subject_wise=True)


