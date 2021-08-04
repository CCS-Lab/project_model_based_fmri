mbfmri.core package
===================

Top wrapping function to run general model-based fMRI analysis.

1. process fMRI & behavioral data to generate multi-voxel bold signals and latent process signals
2. load processed signals.

Then 

**MVPA approach**: fit MVPA models and interprete the models to make a brain map.
**GLM approach**:

mbfmri.core.engine.run\_mbfmri
---------------------------------------------------------------

.. automodule:: mbfmri.core.engine
   :members:
   :undoc-members:
   :show-inheritance:



.. toctree::
   :maxdepth: 1
   :glob:
   :caption: By approach

   mbfmri.core.glm.rst
   mbfmri.core.mvpa.rst