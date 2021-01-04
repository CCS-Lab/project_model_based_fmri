<p align="center">
  <img src="https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/mbmvpa_logo.png" width="1000px">
</p>

# MB-MVPA 

Yedarm Seong: mybirth0407@gmail.com<br>
CheolJun Cho: cjfwndnsl@gmail.com<br>

**MB-MVPA** is a unified Python fMRI analysis tool to find a brain implementation of a latent behavioral state.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*. MB-MVPA provides simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]), and run model-based fMRI analysis using MVPA. 

The basic frameworks of model-based fMRI by O'Doherty et al. (2007)[2] consists of following steps.

1) Computational modeling of subjects' behaviors
2) Extraction & time series generation for state values in the model (a.k.a latent process)
3) Relate latent process with task-fMRI time series data

In **MB_MVPA**, GLM in prevailing  massive univariate approach is replaced with multi-voxel pattern analysis (MVPA) models and has the following two major differences. First, MVPA regression models predict cognitive process directly from brain activations, so enabling acquisition of *reverse inference* model denoted by Poldrack (2006)[3]. Second, the brain activation pattern correlated with the latent process is obtained by interpreting trained MVPA regression models.

MB-MVPA supports Python 3.6 or above and relies on NumPy, NiLearn, hBayesDM, py-glmnet, and tensorflow (version)

### Features

1. MB-MVPA is based on MVPA regression model.
2. MB-MVPA is flexible as it allows various MVPA models plugged in.
3. MB-MVPA is free of analytic hierarhy (e.g. first-level anal. or second-level anal.).

## Resources

- [**Developer Guides**](https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/docs/source/dev-guide.rst)
- [**Getting started**](TODO) 
- [**Documentation**](TODO) 
- [**Bug reports**](TODO) 

## Example data and code

Mixed gamble task by Tom et al. 2007 retrieved from https://openneuro.org/datasets/ds000005/versions/00001<br>
[ipynb notebook link](https://nbviewer.jupyter.org/gist/mybirth0407/58c2f854a8b8790acfb525abedd92571#file-tom_mvpa_model_based_fmri-ipynb) (only viewer)

Delayed discount task by Piva et al. 2019 retrieved from https://doi.org/10.18112/openneuro.ds001882.v1.0.5<br>
notebook: TBU

## References
[1] Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing Neurocomputational Mechanisms of Reinforcement Learning and Decision-Making With the hBayesDM Package. Computational Psychiatry, 1(Figure 1), 24–57. https://doi.org/10.1162/cpsy_a_00002

[2] O’Doherty, J. P., Hampton, A., & Kim, H. (2007). Model-based fMRI and its application to reward learning and decision making. Annals of the New York Academy of Sciences, 1104, 35–53. https://doi.org/10.1196/annals.1390.022

[3] Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? Trends in Cognitive Sciences, 10(2), 59–63. https://doi.org/10.1016/j.tics.2005.12.004
