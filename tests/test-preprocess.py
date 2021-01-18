from mbmvpa.preprocessing.preprocess import DataPreprocessor
from pathlib import Path
import pdb

root = Path('tests/test_example')
dm_model = 'ra_prospect'

def test_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row

def test_filter(row):
    # include all trial data
    return True

def test_latent(row, param_dict):
    ## calculate subjectives utility for choosing Gamble over Safe option
    ## prospect theory with loss aversion and risk aversion is adopted
    modulation = (row["gain"] ** param_dict["rho"]) - (param_dict["lambda"] * (row["loss"] ** param_dict["rho"]))
    row["modulation"] = modulation
    return row


preprocessor = DataPreprocessor(bids_layout=root,
                               adjust_function=test_adjust,
                               filter_function=test_filter,
                               latent_function=test_latent,
                               dm_model=dm_model)
preprocessor.summary()
preprocessor.preprocess(overwrite=True,nchain=2,nwarmup=50,niter=200)

print("TEST PASS!")

