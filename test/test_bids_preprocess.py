import model_based_mvpa as mbmvpa
from model_based_mvpa.preprocessing.bids import *

print(time.strftime('%c', time.localtime(time.time())))
root = '/data2/project_model_based_fmri/piva_dd/'
save_path = '/data2/project_modelbasedMVPA/temp/'
mask_path = '/data2/project_model_based_fmri/ds000005/derivatives/fmriprep/mask/'
bids_preprocess(root,save_path, mask_path)
print(time.strftime('%c', time.localtime(time.time())))
