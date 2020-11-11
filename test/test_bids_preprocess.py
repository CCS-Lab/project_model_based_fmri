import model_based_mvpa as mbmvpa
from model_based_mvpa.preprocessing.bids import *

print(time.strftime('%c', time.localtime(time.time())))
root = '/data2/project_model_based_fmri/piva_dd/'
save_path = '/data2/project_modelbasedMVPA/temp_2/'
mask_path = None
bids_preprocess(root,save_path, mask_path,zoom = (4,4,4))
print(time.strftime('%c', time.localtime(time.time())))
