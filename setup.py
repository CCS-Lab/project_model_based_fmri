#!/usr/bin/env python3
import os
import sys
import subprocess
from setuptools import setup, find_packages
from pathlib import Path


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


PATH_ROOT = Path(__file__).absolute().parent


MAJOR = 0
MINOR = 0
MICRO = 3
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
VERSION += '' if ISRELEASED else '.9000'


DESC = 'Python library for fMRI multivoxel pattern analysis'
# with open('README.rst', 'r', encoding='utf-8') as f:
#     LONG_DESC = f.read()
# LONG_DESC_TYPE = 'text/x-rst'
AUTHOR = 'CCSLab'
AUTHOR_EMAIL = 'ccslab.snu@gmail.com'
URL = 'https://github.com/CCS-Lab/project_model_based_fmri '
LICENSE = 'GPLv3'
CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

setup(
    name='model based mvpa',
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESC,
    # long_description=LONG_DESC,
    # long_description_content_type=LONG_DESC_TYPE,
    python_requires='>=3.6',
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=[
        'bids',
        'nilearn',
        'nibabel',
        'numpy',
        'sklearn',
        'scikit-image',
        'pandas',
        'pystan',
        'hbayesdm',
        'matplotlib',
        'tensorflow',
        'glmnet'
    ],
    zip_safe=False,
    include_package_data=True,
)
