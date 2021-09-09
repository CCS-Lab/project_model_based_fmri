from pathlib import Path

bids_layout = "/home/cheoljun/mbmvpa_paper_writing/tutorial/mini_bornstein2017"

from mbfmri.core.engine import run_mbfmri

_ = run_mbfmri(
               ### To identify, load, and save data
               bids_layout=bids_layout,             # data path - the root for entire BIDS layout 
               task_name='multiarmedbandit',        # identifier for task (BIDS)
               subjects='all',                      # default: 'all' - load all subjects in the layout.
                                                      # could be a list of subject IDs (string) (e.g., ['sub-01', 'sub-02'])
               sessions = 'all',                    # default: 'all', could be a list of sessions.
                                                      # should add example. refer to BIDSDataLoader ('01' or 'ses-01'?)
               #report_path=report_path,             # path for saving outputs (fit reports, brainmap, pearsonr, mse, raw_result)
                                                      # [??] (why "report path" here? not saving path? what's the difference? where is 'save path?')
                                                      # [CJ] (I used "save_path" for indicating the path where the "mbmvpa" derivative will be saved.
                                                      #       It will be set as ROOT/derivatives as default, but user can designate it by "save_path."
                                                      #       Please refer to mbfmri/utils/bids_utils.py (line 314~324).)
    
               feature_name='zoom2rgrout',          # Name for indicating preprocessed feature (default: "unnamed") - to distinguish voxel feature data generated from different configurations.
                                                      # (e.g. the preprocessed file will be saved as: "sub-01_task-learn_desc-zoom2rgrout_voxelfeature.npy" ) 
                                                      # Redundant preprocessing step could be skipped and avoided when the files with the same feature name already exists.
                                                      # But it could be overridden by setting "overwrite = True"
    

               confounds=["trans_x", "trans_y",     # list of confounds (including motion regressors)
                          "trans_z", "rot_x",
                          "rot_y", "rot_z"],    
    
               ### To run computational modeling (use hBayesDM) and make latent process signal
               dm_model= 'banditNarm_lapse_decay',  # computational model
               process_name='PEchosen',             # identifier for target latent process
               refit_compmodel=True,                # indicate if refitting comp. model is required
               n_core=4,                            # number of core for multi-processing in hBayesDM    

    
               ### For fMRI analysis
               analysis='mvpa',                     # name of analysis ('mvpa' or 'glm', default: 'mvpa')
               mvpa_model='elasticnet',             # (ONLY for MVPA) which kind of MVPA model will be used ('elasticnet', 'mlp', or 'cnn')
               method='5-fold',                     # (ONLY for MVPA) type of cross-validation

               n_thread=4,                          # number of thread for multi-threading in generating voxel features
               
    
               ### others
               overwrite=True,                      # indicate if re-generate voxel feaures and latent process and they should be overwritten. (not related to re-fitting hBayesDM)

              )