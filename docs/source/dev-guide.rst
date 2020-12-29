Developer's Guide
=================

Setup
-----

First, you should clone the GitHub repository and checkout the develop branch.

.. code:: bash

    git clone github.com/CCS-Lab/project_model_based_fmri.git
    cd project_model_based_fmri

To manage a virtual environment for development, we uses `poetry`_. Pipfile on
the repository describes which packages to install for a virtual environment.
You can create a virtual environment by running the command below:

.. _poetry:
   https://github.com/python-poetry/poetry

.. code:: bash

    # Install the virtual environment defined in the pyproject.toml
    poetry install

Now you can activate the installed virtual environment. Code blocks on later
sections assume this environment activated.

.. code:: bash

    # Activate the virtual environment
    poetry shell
    python setup.py install
