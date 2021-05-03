<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/mbmvpa_logo.png" width="1000px">
</p>

# MB-MVPA 

<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/mbmvpa_diagram.png" width="1000px">
</p>

**MB-MVPA** is a unified Python fMRI analysis tool on task-based fMRI data to find a brain implementation of a latent behavioral state.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*. MB-MVPA provides simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]), and run model-based fMRI analysis using MVPA. To [install](#Installation).

The basic framework of model-based fMRI by O'Doherty et al. (2007)[2] consists of the following steps.

1) Computational modeling of subjects' behaviors
2) Extraction & time series generation for state values in the model (a.k.a latent process)
3) Relate latent process with task-fMRI time series data

**MB-MVPA** replaced GLM in the prevailing  massive univariate approach with MVPA regression models and has the following two major differences. First, MVPA regression models predict cognitive process directly from brain activations, so enabling acquisition of *reverse inference* model denoted by Poldrack (2006)[3]. Second, instead of mapping statistical significance, the brain activation pattern correlated with the latent process is obtained by interpreting trained MVPA regression models.

**MB-MVPA** supports Python 3.6 or above and relies on [NiLearn](https://github.com/nilearn/nilearn), [hBayesDM](https://github.com/CCS-Lab/hBayesDM/tree/develop/Python), [py-glmnet](https://github.com/civisanalytics/python-glmnet), and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras?hl=ko)(tested on v2.4.0).

### Features

- MB-MVPA is based on MVPA regression model.
- MB-MVPA is flexible as it allows various MVPA models plugged in.
- MB-MVPA is free of analytic hierarchy (e.g. first-level anal. or second-level anal.).

## Computational modeling

Computational modeling is done by wrapping up [hBayesDM](https://github.com/CCS-Lab/hBayesDM/tree/develop/Python) package by Ahn et al. (2017)[1]. Please refer to the [model list](~~) to check the available models and latent processes. Even the model you are looking for is not in the list, you can still conduct the analysis with some manipulations. In this case, please follow use cases in the tutorials, which match best your situation.

## MVPA model

The MVPA models for regressing voxel features against the target latent process can be implemented as any kinds of machine learning models for regression. The package support readily implemented a linear model (ElasticNet), and deep learning models (multi-layer perceptron (MLP) and convolutional neural network (CNN)). 

|model|example|
| :---: | :---: |
|ElasticNet|[link]()|
|MLP|[link]()|
|CNN|[link]()|



You can also plug-in your own models for MB-MVPA analysis, so please refer to the [developer guide](~~). Be aware that the input dimension is likely to be high so training models would require a lot of computing resources including cores, memories and time. 

## Example

The following code is the simplest case of using the package. You can find the detail and other cases in the tutorials.

``` python
from mbmvpa.core.engine import run_mbmvpa

_ = run_mbmvpa(bids_layout='tutorial_data/ccsl_prl',
               dm_model='prl_fictitious_rp_woa',
               task_name='prl',
               process_name='PEchosen')
```


## Input data

<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/mbmvpa_input_layout.png" width="1000px">
</p>


Before using MB-MVPA, raw fMRI data should be primarily processed by conventional fMRI preprocessing pipeline. *recommend to use* [*fmriprep*](https://fmriprep.org/en/stable/) Then, the preprocessed fMRI data are required to be formatted as [**BIDS**](https://bids-specification.readthedocs.io/en/stable/) layout.

MB-MVPA also needs mask images for ROI masking. We recommend to download forward and backward probability maps from [**Neurosynth**](https://neurosynth.org/). Then the MB-MVPA will integrate them into a single mask file. If not provided, the MNI 152 mask will be used instead. Please place mask files under *BIDS_ROOT/derivatives/fmriprep/masks*.

Ex.

```
{BIDS_ROOT}/derivatives/fmriprep/masks/ -reward_association-test_z_FDR_0.01.nii.gz
                                       |-reward_uniformity-test_z_FDR_0.01.nii.gz
                                         ...
                                       |-loss_association-test_z_FDR_0.01.nii.gz
```

## Installation

TODO. It would tenatatively be the below pip command.

``` bash
pip install mb-mvpa
```

## Resources
- [**Documentation**](TODO) 
- [**Developer Guides**](https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/docs/source/dev-guide.rst)


## References
[1] Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing Neurocomputational Mechanisms of Reinforcement Learning and Decision-Making With the hBayesDM Package. Computational Psychiatry, 1(Figure 1), 24–57. https://doi.org/10.1162/cpsy_a_00002

[2] O’Doherty, J. P., Hampton, A., & Kim, H. (2007). Model-based fMRI and its application to reward learning and decision making. Annals of the New York Academy of Sciences, 1104, 35–53. https://doi.org/10.1196/annals.1390.022

[3] Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? Trends in Cognitive Sciences, 10(2), 59–63. https://doi.org/10.1016/j.tics.2005.12.004

## Developed by
CheolJun Cho: cjfwndnsl@gmail.com<br>
Yedarm Seong: mybirth0407@gmail.com<br>
