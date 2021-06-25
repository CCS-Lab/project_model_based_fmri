#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.03

from mbfmri.core.glm import run_mbglm, GLM
from mbfmri.core.mvpa import run_mbmvpa, MBMVPA

GLM = GLM
MBMVPA = MBMVPA
run_mbglm = run_mbglm
run_mbmvpa = run_mbmvpa

valid_analysis = ['mvpa','glm']

def run_mbfmri(analysis='mvpa',
              **kwargs):
    if analysis.lower() == 'mvpa':
        run_mbmvpa(**kwargs)
    elif analysis.lower() == 'glm':
        run_mbglm(**kwargs)
    else:
        raise ValueError(f'ERROR: please enter valid analysis type-{valid_analysis}')
        