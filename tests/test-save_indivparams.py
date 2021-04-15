import hbayesdm.models
import os, importlib
import pandas as pd

# 'bart_ewmv' ?? 
# 'pst_Q' ??


model_list = [f for f in dir(hbayesdm.models) if f[0] != '_']
model_list = [f for f in model_list if 'choiceRT' not in f]
model_list.sort()

exclude = ['bart_ewmv','pst_Q','wcs_sql','task2AFC_sdt','alt_delta','alt_gamma','cgt_cm',
          'choiceRT_ddm','choiceRT_ddm_single','choiceRT_lba','choiceRT_lba_single',
          'dbdm_prob_weight']
model_list = [m for m in model_list if m not in exclude]

for dm_model in model_list:
    individual_params_path = 'indiv_params/'+f'{dm_model}.tsv'
    if os.path.exists(individual_params_path):
        continue
    #latent_process_functions = modelling_module.latent_process_functions
    model = getattr(
            hbayesdm.models, dm_model)(
                data='example',
                ncore=4,
                vb=False,
                )
    
    individual_params = pd.DataFrame(model.all_ind_pars)
    individual_params.index.name = "subjID"
    individual_params = individual_params.reset_index()
    individual_params.to_csv(individual_params_path,
                                     sep="\t", index=False)
    
print("TEST PASS!")

model_list = [f for f in dir(hbayesdm.models) if f[0] != '_']
model_list = [f for f in model_list if 'choiceRT' not in f]
model_list.sort()


for dm_model in model_list:
    individual_params_path = 'indiv_params/'+f'{dm_model}.tsv'
    if os.path.exists(individual_params_path):
        continue
    #latent_process_functions = modelling_module.latent_process_functions
    model = getattr(
            hbayesdm.models, dm_model)(
                data='example',
                ncore=4,
                vb=False,
                )
    
    individual_params = pd.DataFrame(model.all_ind_pars)
    individual_params.index.name = "subjID"
    individual_params = individual_params.reset_index()
    individual_params.to_csv(individual_params_path,
                                     sep="\t", index=False)