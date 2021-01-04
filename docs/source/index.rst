.. MB-MVPA documentation master file, created by
   sphinx-quickstart on Tue Dec 15 11:06:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====
.. image:: https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/mbmvpa_logo.png
   :alt: mb-mvpa logo
   :align: center

**MB-MVPA** is a unified Python fMRI analysis tool to find a brain implementation of a latent behavioral state.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*. MB-MVPA provides simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]_), and run model-based fMRI analysis using MVPA. 

The basic frameworks of model-based fMRI by O'Doherty et al. (2007)[2]_ is done by following steps.

1) Computational modeling of subjects' behaviors
2) Extraction & time series generation for state values in the model (a.k.a latent process)
3) Relate latent process with task-fMRI time series data

The traditional approach in 3. is employing Generalized Linear Model (GLM), for example, in Statistical Parametric Mapping (SPM)[3]_, by fitting a linear model with the latent process against each voxel's values. In the standard GLM approach, latent process and other confound factors are used as independent variables to predict activation of a single voxel. Then the statistical significance (e.g., score of one sample T-test) of the coefficient for the latent process is regarded as quantification of correlation between voxel-activation and the latent process. This analysis is done respectively for each voxel in the region of interest (a massive univariate approach), so the entire activation pattern map is obtained by aggregating scores. 

.. image:: https://github.com/CCS-Lab/project_model_based_fmri/blob/dev0/images/framework_comp.png
   :alt: frameworks comparison
   :align: center

In **MB_MVPA**, GLM is replaced with multi-voxel pattern analysis (MVPA) and has the following two major differences from the traditional one. First, the direction of regression is the opposite, which means activations of voxels are used as independent variables to calculate the latent process. Therefore, MVPA regression employed here predicts cognitive process directly from brain activations, so enabling acquisition of *reverse inference* model denoted by Poldrack (2006)[4]_. Second, the brain activation pattern correlated with the latent process is obtained by interpreting trained MVPA regression models. If a linear model is used, the interpretation is simple as directly regarding model coefficients as the activation pattern, in which each coefficient can be 1-to-1 correspondently mapped to each voxel due to the characteristic of the linear model. However, when it comes to a non-linear model, it is not a simple problem in terms of both the theoretical basis of interpretation and practical issues in calculating it. As models are less likely to have a self-explicative structure, such as having a self-attentive layer in the input space, additive feature attribution methods in AI model interpretation literature (e.g. Lundberg et al. (2017)[5]_) can be applied to interpret non-linear MVPA models as well. But it requires more studies on adopting this approach to fMRI analysis. Here, to make it simple as possible and agnostic to input sample space, the model is regarded as linear so that the model output of feeding one-hot indicator vector is regarded as each voxel's attribution.




MB-MVPA supports Python 3.6 or above and relies on NumPy, NiLearn, hBayesDM, py-glmnet, and tensorflow (version)

Features
--------

1. MB-MVPA is based on MVPA regression model.
2. MB-MVPA is flexible as it allows various MVPA models plugged in.
3. MB-MVPA is free of analytic hierarhy (e.g. first-level anal. or second-level anal.).

Content
-------

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Contents:
   
   dev-guide.rst
   
.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Examples

   jupyter_examples/*
   
.. toctree::
   :maxdepth: 3
   :glob:
   :caption: Modules:
   
   modules

References
----------

..[1] Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing Neurocomputational Mechanisms of Reinforcement Learning and Decision-Making With the hBayesDM Package. Computational Psychiatry, 1(Figure 1), 24–57. https://doi.org/10.1162/cpsy_a_00002

..[2] O’Doherty, J. P., Hampton, A., & Kim, H. (2007). Model-based fMRI and its application to reward learning and decision making. Annals of the New York Academy of Sciences, 1104, 35–53. https://doi.org/10.1196/annals.1390.022

..[3] Friston, K. J. (2004). Statistical parametric mapping. Practical Neurology, 4(6), 350–355. https://doi.org/10.1111/j.1474-7766.2004.00266.x

..[4] Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging data? Trends in Cognitive Sciences, 10(2), 59–63. https://doi.org/10.1016/j.tics.2005.12.004

..[5] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 2017-Decem(Section 2), 4766–4775.


===================================

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
