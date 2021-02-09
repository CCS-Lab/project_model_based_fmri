from time import perf_counter
from mbmvpa.data.loader import BIDSDataLoader

s = perf_counter()

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/ds000005"

loader = BIDSDataLoader(layout=root)
X,y = loader.get_total_data()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

voxel_mask = loader.get_voxel_mask()

from mbmvpa.models.mvpa_mlp import MLP

MVPA_model = MLP(X=X,
                y=y,
                n_patience=50,
                n_repeat=30,
                voxel_mask = voxel_mask)



s = perf_counter()

coeffs = MVPA_model.run()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

s = perf_counter()

sham_errors = MVPA_model.sham()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")

img = MVPA_model.image(save_path='.', task_name='example')