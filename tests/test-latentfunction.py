import hbayesdm
import hbayesdm.models
import os, importlib
import pandas as pd
from pathlib import Path

example_data_path = hbayesdm.base.PATH_EXTDATA 
indiv_param_path = 'tests/indiv_params'
# 'bart_ewmv' ?? 
# 'pst_Q' ??
# 'wcs_sql' ??

exclude = ['bart_ewmv','pst_Q','wcs_sql','task2AFC_sdt','alt_delta','alt_gamma','cgt_cm',
          'choiceRT_ddm','choiceRT_ddm_single','choiceRT_lba','choiceRT_lba_single']

hbayesdm_model_list = [f for f in dir(hbayesdm.models) if f[0] != '_']
model_list = [f.split('.tsv')[0] for f in os.listdir(indiv_param_path)]
model_list = [m for m in model_list if m not in exclude]
model_list = [m for m in model_list if m != '']
model_list = [m for m in model_list if m in hbayesdm_model_list]
model_list.sort()

ncore = 16

def _process_indiv_params(individual_params):
    if type(individual_params) == str\
        or type(individual_params) == type(Path()):
        try:
            individual_params = pd.read_table(individual_params)
            #print("INFO: individual parameters are loaded")
        except:
            individual_params = None
        
        return individual_params
    elif type(individual_params) == pd.DataFrame:
        return individual_params
    else:
        return None

for dm_model in model_list:
    print("testing... "+dm_model)
    individual_params_path = os.path.join(indiv_param_path,f'{dm_model}.tsv')
    individual_params =_process_indiv_params(individual_params_path)
    modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
    modelling_module = importlib.import_module(modelling_module)
    python_model = modelling_module.ComputationalModel("")
    task_name = dm_model.split('_')[0]
    example_task_data_path  = os.path.join(example_data_path, f'{task_name}_exampleData.txt')
    df_events = pd.read_table(example_task_data_path, sep='\t')
    for subjID in individual_params['subjID'].unique():
        param_dict = {k:d.item() for k, d in dict(individual_params[individual_params['subjID']==subjID]).items()}
        _ = python_model._set_latent_process(df_events.copy(),param_dict)
        
print("TEST PASS!")