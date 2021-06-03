__all__ = ["data", "models", "preprocessing", "utils","core"]

import importlib
for module in __all__:
    importlib.import_module('mbfmri.'+module)
    
__version__ = "0.4.0" #TODO automatically retrieve it.
