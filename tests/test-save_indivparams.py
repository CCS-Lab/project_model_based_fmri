import hbayesdm.models
import os, importlib
import pandas as pd

# 'bart_ewmv' ?? 
# 'pst_Q' ??

exclude = ['template.py','__init__.py','base_model.py', 'bart_ewmv']
model_list = os.listdir('../mbmvpa/preprocessing/computational_modeling')
model_list = [f for f in model_list if f not in exclude]
model_list = [f for f in model_list if f[0] != '.']
model_list = [f for f in model_list if f[:2] != '__']
model_list = [f.split('.py')[0] for f in model_list]
#model_list = ['gng_m4','gng_m1','gng_m2','gng_m3','bart_par4','bart_ewmv']
#model_list = ['dbdm_prob_weight']
ncore = 4

for dm_model in model_list:
    modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
    modelling_module = importlib.import_module(modelling_module)
    _ = modelling_module.ComputationalModel("")
    #latent_process_functions = modelling_module.latent_process_functions
    model = getattr(
            hbayesdm.models, dm_model)(
                data='example',
                ncore=ncore,
                vb=False,
                nwarmup=30,
                niter=300,
                nchain=1)
    
    individual_params = pd.DataFrame(model.all_ind_pars)
    individual_params.index.name = "subjID"
    individual_params = individual_params.reset_index()
    individual_params_path = 'indiv_params/'+f'{dm_model}.tsv'
    individual_params.to_csv(individual_params_path,
                                     sep="\t", index=False)
    
print("TEST PASS!")