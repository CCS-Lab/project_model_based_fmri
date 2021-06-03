from mbfmri.core.glm import GLM
from pathlib import Path

bids_layout = "tutorial_data/ccsl_prl"
report_path = "tutorial_report"
Path(report_path).mkdir(exist_ok=True)



glm =  GLM(bids_layout,
           task_name='rpl',
           process_name='PEchosen',
           space_name=None,
           smoothing_fwhm=6,
           confounds=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
           glm_save_path=report_path)

glm.run()