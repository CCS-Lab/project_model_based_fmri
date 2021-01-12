from pathlib import Path 
import json
import mbmvpa

version = mbmvpa.__version__

def make_mbmvpa_description(mbmvpa_root,
                            bids_version,
                            code_url="", 
                            how_to_acknowledge="", 
                            license=""):
    dataset_description= {
                        "Name": "MB-MVPA - Model based MVPA",
                        "BIDSVersion": bids_version,
                        "PipelineDescription": {
                            "Name": "MB-MVPA",
                            "Version": mbmvpa.__version__,
                            "CodeURL": code_url
                        },
                        "CodeURL": code_url,
                        "HowToAcknowledge": how_to_acknowledge,
                        "License": license
                    }
    
    with open(Path(root)/'dataset_description.json', 'w') as f:
        json.dump(dataset_description, f)
        
    return dataset_description
