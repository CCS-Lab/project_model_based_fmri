from mbmvpa.core.glm import run_mbglm
from pathlib import Path

#bids_layout = "tutorial_data/ccsl_prl"
bids_layout = "tests/test_example"
report_path = "tests/test_report"
Path(report_path).mkdir(exist_ok=True)


def test_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row


run_mbglm(report_path=report_path,
          bids_layout=bids_layout,
          dm_model='ra_prospect',
          task_name='mixedgamblestask',
          process_name='SUgamble',
          adjust_function=test_adjust,
          overwrite=False,
          overwrite_latent_process=True,
          refit_compmodel=False)
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