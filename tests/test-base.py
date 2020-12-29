from mbmvpa.preprocessing.bids import bids_preprocess
from mbmvpa.preprocessing.events import events_preprocess
from mbmvpa.data.loader import prepare_dataset
from mbmvpa.utils.example_utils import load_example_data

from mbmvpa.utils.coef2map import get_map
from time import perf_counter
from mbmvpa.models.regressor import penalized_linear_regression, mlp_regression, elasticnet