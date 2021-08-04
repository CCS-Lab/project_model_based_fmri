# project_model_based_fmri
<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/main/images/flowchart_temp.png" width="1000px">
</p>

**MBfMRI** is a unified Python fMRI analysis tool on task-based fMRI data to find a brain implementation of a latent behavioral state.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*. MBfMRI provides simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]), and run model-based fMRI analysis using MVPA. To [install](#Installation).

The basic framework of model-based fMRI by O'Doherty et al. (2007)[2] consists of the following steps.

1) Computational modeling of subjects' behaviors
2) Extraction & time series generation for state values in the model (a.k.a latent process)
3) Relate latent process with task-fMRI time series data

Upon the prevailing massive univariate approach based on GLM, **MBfMRI** extends the framework by adopting MVPA regression models. The MVPA approach (model-based MVPA) has the following two  differences. First, MVPA regression models predict cognitive process directly from brain activations, so enabling acquisition of *reverse inference* model denoted by Poldrack (2006)[3]. Second, instead of mapping statistical significance, the brain activation pattern correlated with the latent process is obtained by interpreting trained MVPA regression models.

**MBfMRI** supports Python 3.6 or above and relies on [NiLearn](https://github.com/nilearn/nilearn), [hBayesDM](https://github.com/CCS-Lab/hBayesDM/tree/develop/Python), [py-glmnet](https://github.com/civisanalytics/python-glmnet), and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras?hl=ko)(tested on v2.4.0).

### Features of model-based MVPA

- Model-based MVPA is based on MVPA regression model.
- Model-based MVPA is flexible as it allows various MVPA models plugged in.
- Model-based MVPA is free of analytic hierarchy (e.g. first-level anal. or second-level anal.).

The package provides previous GLM approach as well.



# Install

```
git clone https://github.com/CCS-Lab/project_model_based_fmri.git
cd project_model_based_fmri
poetry install
poetry shell
python setup.py install
```

# Tutorial

Download the example data from [here](https://drive.google.com/file/d/1nmHwyxgrCfMQ3EhDhdFb3BwToEzMArqN/view?usp=sharing). Download it and unzip it under *tutorials*.<br>
See the [tutorial](https://github.com/CCS-Lab/project_model_based_fmri/blob/main/tutorials/tutorial.ipynb).
