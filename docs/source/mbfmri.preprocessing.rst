mbfmri.preprocessing
====================
This module is for processing fMRI images and behavioral data. 
For fMRI images (*_bold.nii), `mbfmri.preprocessing.bold.VoxelFeatureGenerator` masks, smoothes, and cleans input images to generate multi-voxel signals.
For behavioral data (*_events.tsv), mbfmri.preprocessing.events.LatentProcessGenerator` fits computational models, selects best model, extracts latent process and generate latent process signals. 

The outputfiles will be saved in new BIDS derivatives layout (named as *MB-MVPA*).

The below figure is an example of generated signals.

.. image:: https://raw.githubusercontent.com/CCS-Lab/project_model_based_fmri/main/images/example_signals.png
   :alt: example
   :align: center

.. centered:: |version|

Submodules
----------

mbfmri.preprocessing.bold module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: mbfmri.preprocessing.bold
   :members:
   :undoc-members:
   :show-inheritance:

Atlas-ROIs information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::

   roi_info
   
mbfmri.preprocessing.events module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: mbfmri.preprocessing.events
   :members:
   :undoc-members:
   :show-inheritance:
