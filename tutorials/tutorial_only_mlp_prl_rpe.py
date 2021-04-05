from time import perf_counter
from mbmvpa.models.mvpa_general import MVPA_CV
from mbmvpa.models.tf_mlp import MVPA_MLP
from mbmvpa.utils.report import build_base_report_functions
from mbmvpa.data.loader import BIDSDataLoader
from pathlib import Path

#root = load_example_data("tom")
root = "/data2/project_modelbasedMVPA/PRL"
report_path = "ccsl_prl"
task_name = "prl"
process_name = "rpe"

Path(report_path).mkdir(exist_ok=True)
'''
subjects = ['01','02','03','04','05','06',
            '07','08','09','10', '11', '12',
            ]
'''
subjects = None
loader = BIDSDataLoader(layout=root, process_name=process_name, subjects=subjects, normalizer="minmax")
X_dict,y_dict = loader.get_data(subject_wise=True)
voxel_mask = loader.get_voxel_mask()

input_shape = X_dict[list(X_dict.keys())[0]].shape[1]

model = MVPA_MLP(input_shape,
                 layer_dims=[512, 512],
                 activation="sigmoid",
                 activation_output="linear",
                 dropout_rate=0.5,
                 val_ratio=0.2,
                 optimizer="adam",
                 loss="mse",
                 learning_rate=0.001,
                 n_epoch = 80,
                 n_patience = 15,
                 n_batch = 64,
                 n_sample = 100000,
                 use_bias = True,
                 use_bipolar_balancing = False)



report_function_dict = build_base_report_functions(voxel_mask,
                                                     task_name=task_name,
                                                     map_type='z',
                                                     sigma=1
                                                     )


model_cv = MVPA_CV(X_dict,
                    y_dict,
                    model,
                    model_param_dict={},
                    method='5-fold',
                    n_cv_repeat=5,
                    cv_save=True,
                    cv_save_path=report_path,
                    task_name=task_name,
                    report_function_dict=report_function_dict)


s = perf_counter()
model_cv.run()

print(f"elapsed time: {(perf_counter()-s) / 60:.2f} minutes")
