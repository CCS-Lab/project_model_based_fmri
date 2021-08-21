# project_model_based_fmri
<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/main/images/flowchart_all.png" width="1000px">
</p>

**MBfMRI** is a unified Python fMRI analysis tool on task-based fMRI data to investigate brain implementations of latent neurocognitive processes.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*. MBfMRI offers simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]), and to run model-based fMRI analysis using MVPA. To [install](#Installation).

The basic framework of model-based fMRI by O'Doherty et al. (2007)[2] consists of the following steps.

1) Computational modeling of subjects' behaviors
2) Extraction & time series generation for state values in the model (a.k.a latent process)
3) Relate latent process with task-fMRI time series data

Upon the prevailing massive univariate approach based on GLM, **MBfMRI** extends the framework by adopting MVPA regression models. The MVPA approach (model-based MVPA) has two differences compared to the previous approach: first, MVPA regression models predict cognitive process directly from brain activations, so enabling acquisition of *reverse inference* model denoted by Poldrack (2006)[3]; second, instead of being mapped by statistical significance, the brain activation pattern correlated with the latent process is obtained by interpreting trained MVPA regression models.

**MBfMRI** supports Python 3.6 or above and relies on [NiLearn](https://github.com/nilearn/nilearn), [hBayesDM](https://github.com/CCS-Lab/hBayesDM/tree/develop/Python), [py-glmnet](https://github.com/civisanalytics/python-glmnet), and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras?hl=ko)(tested on v2.4.0).

### Features of model-based MVPA

- Model-based MVPA is based on MVPA regression model.
- Model-based MVPA is flexible as it allows various MVPA models plugged in.
- Model-based MVPA is free of analytic hierarchy (e.g. first-level anal. or second-level anal.).

### Models implemented in MBfMRI
- MBfMRI offers [various MVPA models](https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.models.html#models) as well as [a massive univariate approach based on GLM](https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.glm.html).


# Installation

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

## References
[1] Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing Neurocomputational Mechanisms of Reinforcement Learning and Decision-Making With the hBayesDM Package. Computational Psychiatry, 1(Figure 1), 24–57. https://doi.org/10.1162/cpsy_a_00002

[2] O’Doherty, J. P., Hampton, A., & Kim, H. (2007). Model-based fMRI and its application to reward learning and decision making. Annals of the New York Academy of Sciences, 1104, 35–53. https://doi.org/10.1196/annals.1390.022

[3] Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? Trends in Cognitive Sciences, 10(2), 59–63. https://doi.org/10.1016/j.tics.2005.12.004
