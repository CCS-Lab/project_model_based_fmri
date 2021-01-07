from mbmvpa.preprocessing.bids import bids_preprocess
from mbmvpa.data.loader import prepare_dataset
from mbmvpa.utils.example_utils import load_example_data
from mbmvpa.utils.coef2map import get_map

from mbmvpa.models.model import *
from mbmvpa.preprocessing.events import *

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/ds000005"

'''
X, voxel_mask, layout, data_root = bids_preprocess(root, smoothing_fwhm=None, zoom=(2, 2, 2), ncore=2, nthread=4)

def example_tom_preprocess_columns(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row

def example_tom_condition(row):
    # include all trial data
    return True

def example_tom_modulation(row, param_dict):
    ## calculate subjectives utility for choosing Gamble over Safe option
    ## prospect theory with loss aversion and risk aversion is adopted
    modulation = (row["gain"] ** param_dict["rho"]) - (param_dict["lambda"] * (row["loss"] ** param_dict["rho"]))
    row["modulation"] = modulation
    return row

generator = LatentProcessGenerator(root=root,
                                   preprocess=example_tom_preprocess_columns,
                                   condition=example_tom_condition,
                                   modulation=example_tom_modulation,
                                   dm_model='ra_prospect')

y, time_mask = generator.run()
'''

MVPA_model = MLP(root=root)
coefs = MVPA_model.run()
pdb.set_trace()
#get_map(coefs, MVPA_model.voxel_mask, task_name="tom2007_mlp", map_type="z", save_path=".", sigma=1)
img = MVPA_model.image()

