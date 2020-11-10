# Design issue


## 1. Data preprocessing pipeline design 

**issue** : TOO LARGE fMRI data. current method is using on-memory preprocessing and makes a fulle input data matrix (X) at once with naive multiprocessing.

**possible solution** : process fMRI data with partially sequencial processing storing subject-wise preprocessed data.

resulting directory might looks like

```
BIDS root -- derivatives -- fmriprep
                        |-- prepprep -- subj01.npy
                                    |-- subj02.npy
                                           .
                                           .
                                           .
                                    |-- medta_data.json     # meta info e.g. number of subjects, data shape
                                    |-- mask.npy            # ROIs mask
                                           
                          

```
