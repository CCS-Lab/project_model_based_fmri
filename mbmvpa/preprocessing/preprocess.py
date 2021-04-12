from .events import LatentProcessGenerator
from .bold import VoxelFeatureGenerator
import numpy as np

class DataPreprocessor():
    
    def __init__(self,
                  bids_layout,
                  save_path=None,
                  task_name=None,
                  mask_path=None,
                  fmriprep_name='fMRIPrep',
                  bold_suffix='bold',
                  confound_suffix='regressors',
                  mask_threshold=2.58,
                  zoom=(2, 2, 2),
                  smoothing_fwhm=6,
                  standardize=True,
                  confounds=[],
                  high_pass=None,
                  detrend=True,
                  n_core=4,
                  process_name="unnamed",
                  feature_name="unnamed",
                  adjust_function=lambda x: x,
                  filter_function=lambda _: True,
                  latent_function=None,
                  modulation_dfwise=None,
                  dm_model="unnamed",
                  filter_for_modeling=None,
                  individual_params=None,
                  hrf_model="glover",
                  use_duration=False,
                  ignore_original=False,
                  onset_name="onset",
                  duration_name="duration",
                  end_name=None,
                  use_1sec_duration=True):
        
        self.X_generator = VoxelFeatureGenerator(bids_layout=bids_layout,
                                          save_path=save_path,
                                          task_name=task_name,
                                          feature_name=feature_name,
                                          fmriprep_name=fmriprep_name,
                                          mask_path=mask_path,
                                          bold_suffix=bold_suffix,
                                          confound_suffix=confound_suffix,
                                          mask_threshold=mask_threshold,
                                          zoom=zoom,
                                          smoothing_fwhm=smoothing_fwhm,
                                          standardize=standardize,
                                          confounds=confounds,
                                          high_pass=high_pass,
                                          detrend=detrend,
                                          n_thread=n_core,
                                          ignore_original=ignore_original)
        
        self.bids_controller = self.X_generator.bids_controller
        
        self.y_generator = LatentProcessGenerator(bids_layout=bids_layout,
                                          bids_controller=self.bids_controller,
                                          save_path=save_path,
                                          task_name=task_name,
                                          process_name=process_name,
                                          adjust_function=adjust_function,
                                          filter_function=filter_function,
                                          latent_function=latent_function,
                                          modulation_dfwise=modulation_dfwise,
                                          dm_model=dm_model,
                                          filter_for_modeling=filter_for_modeling,
                                          individual_params=individual_params,
                                          hrf_model=hrf_model,
                                          use_duration=use_duration,
                                          n_core=n_core,
                                          onset_name=onset_name,
                                          duration_name=duration_name,
                                          end_name=end_name,
                                          use_1sec_duration=use_1sec_duration)
    
    def summary(self):
        self.bids_controller.summary()
    
    def preprocess(self, 
                   overwrite=False, 
                   process_name=None,
                   feature_name=None, 
                   confounds=None,
                   n_thread=None,
                   dm_model=None, 
                   individual_params=None, 
                   df_events=None, 
                   adjust_function=None, 
                   filter_function=None,
                   skip_modeling=False,
                   **kwargs):
            
        
        self.X_generator.run(overwrite=overwrite,
                           feature_name=feature_name,
                           confounds=confounds,
                           n_thread=n_thread,
                            **kwrags)
        
        if not skip_modeling:
            self.y_generator.set_computational_model(overwrite=overwrite,
                                                    dm_model=dm_model, 
                                                    individual_params=individual_params, 
                                                    df_events=df_events, 
                                                    adjust_function=adjust_function, 
                                                    filter_function=filter_function, 
                                                    **kwargs)
        
        self.y_generator.run(overwrite=overwrite,
                            process_name=process_name)
        
        self.bids_controller.reload()
        
        
    
        
    