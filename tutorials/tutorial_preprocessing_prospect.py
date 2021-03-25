from time import perf_counter
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from mbmvpa.models.mvpa_elasticnet import elasticnet_crossvalidation
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path

#root = load_example_data("tom")

root = "/data2/project_modelbasedMVPA/ds000005"

Path(report_path).mkdir(exist_ok=True)

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
                               #save_path=save_path,
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
