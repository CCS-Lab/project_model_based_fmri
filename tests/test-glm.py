from mbmvpa.core.glm import GLM, run_glm
from pathlib import Path

#bids_layout = "tutorial_data/ccsl_prl"
bids_layout = "tests/test_example"
report_path = "tests/test_report"
Path(report_path).mkdir(exist_ok=True)



'''
glm =  GLM(bids_layout,
           task_name='mixedgamblestask',
           process_name='SUgamble',
           space_name=None,
           smoothing_fwhm=6,
           confounds=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
           glm_save_path=report_path)

glm.run()

'''