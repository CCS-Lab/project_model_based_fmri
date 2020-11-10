# Design issue


## 1. Data preprocessing pipeline design 

**issue** : TOO LARGE fMRI data. current method is using on-memory preprocessing and makes a fulle input data matrix (X) at once with naive multiprocessing.

**possible solution** : process fMRI data with partially sequencial processing storing subject-wise preprocessed data.

resulting directory might looks like

```
BIDS root -- derivatives -- fmriprep
                        |-- prepprep -- subj01.npy          # preprocessed data per subject. data shape : (ses # x run # x time #) x masked voxels #
                                    |-- subj02.npy
                                           .
                                           .
                                           
                                    |-- medta_data.json     # meta info e.g. number of subjects, data shape
                                    |-- mask.npy            # ROIs mask
                                           
                          

```

Then, following should also be considered. 

**Dataloading for model fitting**

**Cross validation : subject level vs agnostic ratio**

**Constrain on max number of data for model fitting**



## 2. Usability

the optimal level of abstraction for user to use the package e.g. use the tool with a single line command by specifying BIDS root, ROI masking, target latent process.

## 3. Package name

## Other

CJC ) suggests to use 8x8x8 mean pooled data if ROI masking info is not given.
