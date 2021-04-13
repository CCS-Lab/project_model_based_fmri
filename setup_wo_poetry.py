#!/usr/bin/env python3
import os
import sys
import subprocess
from setuptools import setup, find_packages
from pathlib import Path


if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")


PATH_ROOT = Path(__file__).absolute().parent

DISTNAME = 'mb-mvpa'

MAJOR = 0
MINOR = 2
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
VERSION += '' if ISRELEASED else '.9000'


DESC = 'Python library for fMRI multi-voxel pattern analysis'
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

install_requires=[
            "hbayesdm<=1.0.1",
            "nibabel>=3.2.0",
            "nilearn==0.7.0",
            "glmnet==2.2.1",
            "pystan>=2.19.1",
            "bids==0.0",
            "scikit-image==0.17.2",
            "matplotlib>=3.3.3",
            "tensorflow>=2.4.0",
            "numpy>=1.18.5",
            "pandas>=1.1.4",
            "scikit-learn==0.24",
            "scipy>=1.5.4",
            "tqdm>=4.0.0",
            "statsmodels==0.12.2",
            "PyYaml==5.4.1",
          ]

setup(
    name='mb-mvpa',
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESC,
    # long_description=LONG_DESC,
    # long_description_content_type=LONG_DESC_TYPE,
    python_requires='>=3.6',
    install_requires=install_requires,
    url=URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
)
