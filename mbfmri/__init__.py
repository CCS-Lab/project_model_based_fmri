__all__ = ["data", "models", "preprocessing", "utils","core"]
__version__ = "0.4.0" #TODO automatically retrieve it.

import importlib
for module in __all__:
    importlib.import_module('mbfmri.'+module)
    

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
