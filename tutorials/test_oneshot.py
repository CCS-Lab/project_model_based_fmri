from mbmvpa.core.engine import MBMVPA

model = MBMVPA(root='tutorial_data/ccsl_prl',
           dm_model='prl_fictitious_rp_woa',
           task='prl',
           process='PEchosen')

_ = model.run()