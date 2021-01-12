from mbmvpa.preprocessing.bold import bold_preprocess
from mbmvpa.data.loader import prepare_dataset
from mbmvpa.utils.example_utils import load_example_data
from mbmvpa.utils.coef2map import get_map

from mbmvpa.models.mvpa_mlp import MLP
from mbmvpa.preprocessing.events import *
import pdb

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/ds000005"

'''
X, voxel_mask, layout, data_root = bold_preprocess(root, smoothing_fwhm=6, zoom=(2, 2, 2), ncore=2, nthread=4)

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
                                   dm_model='ra_prospect',
                                   normalizer='minmax',
                                   individual_params_custom="/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/mvpa/individual_params.tsv"
                                    )

y, time_mask = generator.run()
'''
#pdb.set_trace()



MVPA_model = MLP(root=root,use_bipolar_balancing=False,n_repeat=10,n_epoch=200,n_patience=20,use_default_extractor=False)
coefs = MVPA_model.run()

'''
np.save('coefs_original.npy',coefs)
img = MVPA_model.image(save_path='.', task_name='original')

np.random.shuffle(MVPA_model.y)

coefs = MVPA_model.run()
np.save('coefs_sham.npy', coefs)
img = MVPA_model.image(save_path='.',task_name='sham')


MVPA_model = MLP(root=root,use_bipolar_balancing=True,n_repeat=10,n_epoch=200,n_patience=20,use_default_extractor=False)
coefs = MVPA_model.run()
np.save('coefs_balanced.npy',coefs)
img = MVPA_model.image(save_path='.',task_name='balanced')

np.random.shuffle(MVPA_model.y)

coefs = MVPA_model.run()
np.save('coefs_balanced_sham.npy',coefs)
img = MVPA_model.image(save_path='.',task_name='balanced_sham')
'''