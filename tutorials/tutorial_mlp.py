from time import perf_counter
from mbmvpa.data.loader import BIDSDataLoader
from mbmvpa.preprocessing.preprocess import DataPreprocessor
from scipy import stats
s = perf_counter()

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/ds000005"

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


preprocessor = DataPreprocessor(bids_layout=root,
                               adjust_function=example_adjust,
                               filter_function=example_filter,
                               latent_function=example_latent,
                               dm_model=dm_model,
                               zoom=(1,1,1))

preprocessor.preprocess(overwrite=False,n_core=16)

loader = BIDSDataLoader(layout=root)
X,y = loader.get_total_data()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

voxel_mask = loader.get_voxel_mask()

from mbmvpa.models.mvpa_mlp import MLP

MVPA_model = MLP(X=X,
                y=y,
                layer_dims=[1024,512,256,128],
                activation='sigmoid',
                n_patience=25,
                n_repeat=15,
                voxel_mask = voxel_mask)



s = perf_counter()

coeffs = MVPA_model.run()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

s = perf_counter()

sham_errors = MVPA_model.sham()

print(stats.ttest_ind(MVPA_model._errors, MVPA_model._sham_errors))

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

img = MVPA_model.image(save_path='.', task_name='example')