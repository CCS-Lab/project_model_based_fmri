import model_based_mvpa as mbmvpa
from model_based_mvpa.preprocessing.bids import *

print(time.strftime('%c', time.localtime(time.time())))
root = '/data2/project_model_based_fmri/piva_dd/'
bids_preprocess(root)
print(time.strftime('%c', time.localtime(time.time())))