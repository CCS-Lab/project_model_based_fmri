.. model-based fMRI documentation master file, created by
   sphinx-quickstart on Wed Aug  4 23:43:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

.. image:: https://raw.githubusercontent.com/CCS-Lab/project_model_based_fmri
/main/images/flowchart_all.png
   :alt: flowchart
   :align: center

.. centered:: |version|
----

|**MBfMRI** is a unified Python fMRI analysis tool on task-based fMRI data to
investigate brain implementations of latent neurocognitive processes. MBfMRI
offers simple executable functions to conduct computational modeling (supported
by `hBayesDM <https://doi.org/10.1162/cpsy_a_00002>`_ [1]) and run model-based
fMRI analysis. It is featured by a new analytic framework adopting
*multi-voxel pattern analysis (MVPA)*.

|The analysis is based on the framework of model-based fMRI by `O'Doherty et al.
(2007) <https://doi.org/10.1196/annals.1390.022>`_ [2].

|1) Find best fitting parameters of model to behavioral data
|2) Generate model-based time series by using the best model-fitting parameters
and convolve it with HRF (a.k.a. latent process)
|3) Relate latent process with task-fMRI data

|For the step 3), the prevailing method in model-based fMRI analysis is based on
Generalized Linear Model (GLM), a massive univariate approach followed by
parametric mapping (GLM approach). Upon the GLM approach, **MBfMRI** extends
the framework by adopting MVPA regression models (MVPA approach). The MVPA
approach has the following two differences compared to the previous approach:
(1) MVPA regression models predict cognitive process directly from brain
activations, enabling acquisition of *reverse inference* model denoted by
`Poldrack (2006) <https://doi.org/10.1016/j.tics.2005.12.004>`_ [3]. (2)
instead of mapping statistical significance, the brain activation pattern
correlated with the latent process is obtained by interpreting trained MVPA
regression models. (3) model-based MVPA is flexible as it allows various
MVPA models plugged in and (4) it is free of analytic hierarchy
(e.g. first-level anal. or second-level anal.).

|The specific workflow of the model-based MVPA approach consists of the
following steps.

.. image:: https://raw.githubusercontent.com/CCS-Lab/project_model_based_fmri/
main/images/mbmvpa_workflow.png
   :alt: mbmvpa_workflow
   :align: center

|1) Generate latent process signals by fitting computational models with behavioral
data and convolving time series of latent process with HRF.
|2) Generate multi-voxel signals from preprocessed fMRI images by allowing ROI
masking, zooming spatial resolution, and improving the quality of signals by several
well-established methods (e.g., detrending, high-pass filtering, regressing out confounds).
|3) Train MVPA models by feeding multi-voxel signals as input (X) and latent process
signals as output (y), or target, employing the repeated cross-validation framework.
|4) Interpret the trained MVPA models by *Feature Attribution*  method, and obtain a
3D brain image that quantifies brain activation pattern attributed to predict the
target signals from the multi-voxel signals.

|The package also provides the existing model-based GLM approach. The process shared
with the model-based MVPA approach is **1) Generate latent process signals** to provide
parametric modulation of the target signals. The first-level and second-level analysis
are done by **NiLearn** modules, `FirstLevelModel <https://nilearn.github.io/modules/
generated/nilearn.glm.first_level.FirstLevelModel.html>`_ and `SecondLevelModel <https:
//nilearn.github.io/modules/generated/nilearn.glm.second_level.SecondLevelModel.html>`_
respectively. Please refer to the links if you'd like to know the details about each step.

**MBfMRI** supports Python 3.6 or above and relies on `NiLearn <https://github.com/
nilearn/nilearn>`_, `hBayesDM <https://github.com/CCS-Lab/hBayesDM/tree/develop/
Python>`_, `py-glmnet <https://github.com/civisanalytics/python-glmnet>`_, and
`tensorflow <https://www.tensorflow.org/api_docs/python/tf/keras?hl=ko>`_
(tested on v2.4.0).



Examples
--------
.. code:: python

    from mbfmri.core.engine import run_mbfmri
    import hbayesdm

    _ = run_mbfmri(analysis='mvpa',                     # name of analysis, "mvpa" or "glm"
                   bids_layout='mini_bornstein2017',    # data
                   mvpa_model='elasticnet',             # MVPA model, "mlp" or "cnn" for DNN
                   dm_model= 'banditNarm_lapse_decay',  # computational model
                   feature_name='zoom2rgrout',          # indentifier for processed fMRI data
                   task_name='multiarmedbandit',        # identifier for task
                   process_name='PEchosen',             # identifier for target latent process
                   subjects='all',                      # list of subjects to include
                   method='5-fold',                     # type of cross-validation
                   report_path=report_path,             # save path for reporting results
                   confounds=["trans_x", "trans_y",     # list of confounds to regress out
                              "trans_z", "rot_x",
                              "rot_y", "rot_z"],
                   n_core=4,                            # number of core for multi-processing in hBayesDM
                   n_thread=4,                          # number of thread for multi-threading in generating voxel features
                   overwrite=True,                      # indicate if re-run and overwriting are required
                   refit_compmodel=True,                # indicate if refitting comp. model is required
                  )


Please refer to the documentation of `mbfmri.core.engine.run_mbfmri
<https://project-model-based-fmri.readthedocs.io/en/latest/
mbfmri.core.html#mbfmri.core.engine.run_mbfmri>`_, and you will find links
for the detailed explanation on configuring analysis on the bottom of
`run_mbfmri` docs.
    

Content
-------

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: API Reference

   mbfmri.core.rst
   mbfmri.core.glm.rst
   mbfmri.preprocessing.rst
   mbfmri.data.rst
   mbfmri.models.rst
   mbfmri.utils.explainer.rst
   mbfmri.utils.report.rst
   
References
----------

[1] Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing Neurocomputational
Mechanisms of Reinforcement Learning and Decision-Making With the hBayesDM Package.
Computational Psychiatry, 1(Figure 1), 24–57. https://doi.org/10.1162/cpsy_a_00002

[2] O’Doherty, J. P., Hampton, A., & Kim, H. (2007). Model-based fMRI and its
application to reward learning and decision making. Annals of the New York Academy
of Sciences, 1104, 35–53. https://doi.org/10.1196/annals.1390.022

[3] Poldrack, R. A. (2006). Can cognitive processes be inferred from neuroimaging
data? Trends in Cognitive Sciences, 10(2), 59–63. https://doi.org/10.1016/j.tics.2005.12.004

