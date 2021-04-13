import hbayesdm.models
import os, importlib
import pandas as pd

exclude = ['template.py','__init__.py']
model_list = os.listdir('mbmvpa/preprocessing/computational_modeling')
model_list = [f for f in model_list if f not in exclude]
model_list = [f for f in model_list if f[0] != '.']
model_list = [f for f in model_list if f[:2] != '__']
model_list = [f.split('.py')[0] for f in model_list]
ncore = 4

non_pass = []
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
                niter=100,
                nchain=1)
    
    individual_params = pd.DataFrame(model.all_ind_pars)
    individual_params.index.name = "subjID"
    individual_params = individual_params.reset_index()
    data  = model._TaskModel__raw_data
    data['onset'] = [0]*len(data)
    data['duration'] = [0]*len(data)
    subjID_list = data['subjID'].unique()
    
    for subjID in subjID_list:
        df_events = data[data['subjID']==subjID].copy()
        param_dict = dict(individual_params[individual_params['subjID']==subjID])
        
        try:
            _._set_latent_process(df_events,param_dict)
        except:
                non_pass.append(dm_model)
        '''
        for k, func in latent_process_functions.items():
            try:
                _ = func(df_events,param_dict)
            except:
                non_pass.append(f'{dm_model}.{k}')
        '''
if len(non_pass) ==0:
    print("TEST PASS!")
else:
    print(f"TEST FAILED - {non_pass}")