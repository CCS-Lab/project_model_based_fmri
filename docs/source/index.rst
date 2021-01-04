.. MB-MVPA documentation master file, created by
   sphinx-quickstart on Tue Dec 15 11:06:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

**MB-MVPA** is a unified Python fMRI analysis tool to find a brain implementation of a latent behavioral state.
It combines two fMRI analytic frameworks: *model-based fMRI* and *multi-voxel pattern anlysis (MVPA)*, which is
featured by replacing a massive univariate analysis with MVPA. MB-MVPA provides simple executable functions to conduct 
computational modeling (supported by *hBayesDM*[1]_), and run model-based MVPA. 

The basic frameworks of model-based fMRI by O'Doherty et al. (2007) [2]_

MB-MVPA supports Python 3.6 or above and relies on NumPy, NiLearn, hBayesDM, py-glmnet, and tensorflow (version)

Features
--------

1. MB-MVPA is based on mutli-voxel pattern analysis.
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



===================================

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
