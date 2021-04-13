from mbmvpa.core.engine import run_mbmvpa

_ = run_mbmvpa(bids_layout='tutorial_data/ccsl_prl', # BIDS root path
               dm_model='prl_fictitious_rp_woa', # hBaysDM model name
               feature_name='zoom2', # default name : 'unnamed'
               task_name='prl', 
               process_name='PEnotchosen',
               report_path='result/sample', # debuggin
               n_core=4, # hBayesdm core num
               n_thread=4 # voxel prep. thread num,
               )
