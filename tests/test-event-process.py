from mbmvpa.preprocessing.events import LatentProcessGenerator
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


generator = LatentProcessGenerator(bids_layout=root,
                                   adjust_function=test_adjust,
                                   filter_function=test_filter,
                                   latent_function=test_latent,
                                   dm_model=dm_model)
generator.summary()
generator.set_computational_model(overwrite=True,nchain=2,nwarmup=50,niter=200)

generator = LatentProcessGenerator(bids_layout=root,
                                   adjust_function=test_adjust,
                                   filter_function=test_filter,
                                   latent_function=test_latent,
                                   dm_model=dm_model)
generator.summary()
generator.set_computational_model(overwrite=False)
generator.run(overwrite=True)

'''
boldsignals, time_mask = generator.run(nchain=2,nwarmup=50,niter=200)

df_events = generator._df_events_ready.to_csv(
                root / 'df_events.tsv',
                sep="\t", index=False)

df_events = pd.read_table(root / 'df_events.tsv')

generator = LatentProcessGenerator(root=root,
                                   df_events_custom=df_events)

boldsignals, time_mask = generator.run()

individual_params = root / 'derivatives/fmriprep/mvpa/individual_params.tsv'

generator = LatentProcessGenerator(root=root,
                                   preprocess=test_preprocess,
                                   condition=test_condition,
                                   modulation=test_modulation,
                                   individual_params_custom=individual_params)

boldsignals, time_mask = generator.run()
'''
print("TEST PASS!")

