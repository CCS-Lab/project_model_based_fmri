from pathlib import Path 
import json
import mbmvpa
from ..utils import config # configuration for default names used in the package

version = mbmvpa.__version__

def make_mbmvpa_description(mbmvpa_root,
                            bids_version,
                            code_url="", 
                            how_to_acknowledge="", 
                            license=""):
    dataset_description= {
                        "Name": config.ANAL_NAME,
                        "BIDSVersion": bids_version,
                        "PipelineDescription": {
                            "Name": config.MBMVPA_PIPELINE_NAME,
                            "Version": mbmvpa.__version__
                        }
                    }
    
    with open(Path(mbmvpa_root)/'dataset_description.json', 'w') as f:
        json.dump(dataset_description, f)
        
    return dataset_description

def version_diff(version1, version2):
    version1 = version1.split('.')
    version2 = version2.split('.')
    
    for i in range(len(version1)-len(version2)):
        version2 += [0]
    for i in range(len(version2)-len(version1)):
        version1 += [0]
    
    assert len(version1) == len(version2)
    
    for i,j in zip(version1, version2):
        if i<j:
            return -1
        elif i>j:
            return +1
        else:
            continue
    return 0
        