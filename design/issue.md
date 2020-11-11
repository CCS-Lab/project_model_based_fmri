# Design issue


## 1. Data preprocessing pipeline

**issue** : TOO LARGE fMRI data. current method is using on-memory preprocessing and makes a fulle input data matrix (X) at once with naive multiprocessing.

**possible solution** : process fMRI data with partially sequencial processing, storing subject-wise preprocessed data.

resulting directory might look like

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

**Constraint on max number of data for model fitting**



## 2. Usability

the optimal level of abstraction for user to use the package. e.g. user can use this tool with a single line command by specifying BIDS root, ROI masking, target latent process.

## 3. Package name

_model-based MVPA_ is too long. we need a concise and intuitive name. 

## 4. Drawing activation map

we get repeat N x coefficients from model fitting. coefficients can be readily converted to MNI152 space using ROI masking info. 

as the number of repetition is limited, calculating survival rate of each coefficient is not plausible. therefore, here, we are tentatively using **z map** and **t map**.

**current approach** 


1) z map : calculate z-score using all values in converted MNI152 space coefficients, and average scores along the axis of repetition. 
2) t map : calculate voxel wise one sample t-score of converted MNI152 space coefficients along the axis of repetition.

## Other suggestion

CJC ) use 8x8x8 mean pooled data if ROI masking info is not given.

##  Example data

Delayed discount task by Piva et al. 2019 retrieved from https://doi.org/10.18112/openneuro.ds001882.v1.0.5
