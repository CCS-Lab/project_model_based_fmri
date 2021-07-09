import nibabel as nib
import numpy as np
import nilearn.datasets
from pathlib import Path
from nilearn.image import resample_to_img
import json

FETCH_ATLAS_MODULES = { m.split('fetch_atlas_')[-1]:m for m in dir(nilearn.datasets) if 'fetch_atlas' in m}

                     
                     
ATLAS_ARGS = {'aal': {'version':['SPM12']},
                'destrieux_2009': {'lateralized':[True,False]},
                'harvard_oxford': {'atlas_name': ['cort-maxprob-thr0-1mm',
                                                  'cort-maxprob-thr0-2mm',
                                                  'cort-maxprob-thr25-1mm',
                                                  'cort-maxprob-thr25-2mm',
                                                  'cort-maxprob-thr50-1mm',
                                                  'cort-maxprob-thr50-2mm',
                                                  'sub-maxprob-thr0-1mm',
                                                  'sub-maxprob-thr0-2mm',
                                                  'sub-maxprob-thr25-1mm',
                                                  'sub-maxprob-thr25-2mm',
                                                  'sub-maxprob-thr50-1mm',
                                                  'sub-maxprob-thr50-2mm',
                                                  'cort-prob-1mm',
                                                  'cort-prob-2mm',
                                                  'sub-prob-1mm',
                                                  'sub-prob-2mm']},
                'pauli_2017': {},
                'schaefer_2018': {'n_rois':[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
                'yeo_2011': {'name':['thin_7', 'thick_7', 'thin_17', 'thick_17']}
               }
                  
'''
AVAILABLE_ATLASES = []
for atlas,kwargs in ATLAS_ARGS.items():
    AVAILABLE_ATLASES.append(atlas)
    for key, args in kwargs.items():
        for arg in args:
            AVAILABLE_ATLASES.append((atlas + '_' + str(arg)))
'''
AVAILABLE_ATLASES = ['aal',
                 'aal_SPM12',
                 'destrieux_2009',
                 'destrieux_2009_True',
                 'destrieux_2009_False',
                 'harvard_oxford',
                 'harvard_oxford_cort-maxprob-thr0-1mm',
                 'harvard_oxford_cort-maxprob-thr0-2mm',
                 'harvard_oxford_cort-maxprob-thr25-1mm',
                 'harvard_oxford_cort-maxprob-thr25-2mm',
                 'harvard_oxford_cort-maxprob-thr50-1mm',
                 'harvard_oxford_cort-maxprob-thr50-2mm',
                 'harvard_oxford_sub-maxprob-thr0-1mm',
                 'harvard_oxford_sub-maxprob-thr0-2mm',
                 'harvard_oxford_sub-maxprob-thr25-1mm',
                 'harvard_oxford_sub-maxprob-thr25-2mm',
                 'harvard_oxford_sub-maxprob-thr50-1mm',
                 'harvard_oxford_sub-maxprob-thr50-2mm',
                 'harvard_oxford_cort-prob-1mm',
                 'harvard_oxford_cort-prob-2mm',
                 'harvard_oxford_sub-prob-1mm',
                 'harvard_oxford_sub-prob-2mm',
                 'pauli_2017',
                 'schaefer_2018',
                 'schaefer_2018_100',
                 'schaefer_2018_200',
                 'schaefer_2018_300',
                 'schaefer_2018_400',
                 'schaefer_2018_500',
                 'schaefer_2018_600',
                 'schaefer_2018_700',
                 'schaefer_2018_800',
                 'schaefer_2018_900',
                 'schaefer_2018_1000',
                 'yeo_2011',
                 'yeo_2011_thin_7',
                 'yeo_2011_thick_7',
                 'yeo_2011_thin_17',
                 'yeo_2011_thick_17']

PATH_ROOT = Path(__file__).absolute().parent
PATH_EXTDATA = (PATH_ROOT/'extdata').resolve()
AVAILABLE_ATLAS_LABELS_PATH = PATH_EXTDATA/ 'atlas_label.json'
AVAILABLE_ATLAS_LABELS = json.load(open(AVAILABLE_ATLAS_LABELS_PATH))



def _load_aal(version='SPM12'):
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['aal'])(version=version)
    atlas_map = nib.load(atlas_data.maps)
    atlas_label_mapping = {label:index for label,index in zip(atlas_data.labels,atlas_data.indices)}
    return atlas_map, atlas_label_mapping


def _load_destrieux_2009(lateralized=True):
    lateralized = bool(lateralized)
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['destrieux_2009'])(lateralized=lateralized)
    atlas_map = nib.load(atlas_data.maps)
    atlas_label_mapping = {label.decode('UTF-8'):index for index,label in atlas_data.labels}
    return atlas_map, atlas_label_mapping


def _load_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm'):
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['harvard_oxford'])(atlas_name=atlas_name)
    atlas_map = nib.load(atlas_data.maps)
    atlas_label_mapping = {label:index for index,label in enumerate(atlas_data.labels)}
    return atlas_map, atlas_label_mapping


def _load_pauli_2017():
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['pauli_2017'])(version='det')
    atlas_map = nib.load(atlas_data.maps)
    atlas_label_mapping = {label:index+1 for index,label in enumerate(atlas_data.labels)}
    return atlas_map, atlas_label_mapping

def _load_schaefer_2018(n_rois=100):
    n_rois = int(n_rois)
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['schaefer_2018'])(n_rois=100)
    atlas_map = nib.load(atlas_data.maps)
    atlas_label_mapping = {label.decode('UTF-8'):index+1 for index,label in enumerate(atlas_data.labels)}
    return atlas_map, atlas_label_mapping


def _load_yeo_2011(name='thin_7'):
    assert name in ['thin_7', 'thick_7', 'thin_17', 'thick_17']
    atlas_data = getattr(nilearn.datasets,FETCH_ATLAS_MODULES['yeo_2011'])()
    atlas_map = nib.load(atlas_data[name])
    atlas_label_mapping = {}
    netnum = name.split('_')[-1]
    f = open(atlas_data[f'colors_{netnum}'], 'r')
    while True:
        z = f.readline()
        if z == '':
            break
        index, label = [s for s in z.split(' ') if len(s) >0][:2]
        if index == 0:
            continue
        atlas_label_mapping[label]=index
    f.close()
    
    return atlas_map, atlas_label_mapping
                        
    
   

ATLAS_LOADER = {'aal':_load_aal,
                 'destrieux_2009':_load_destrieux_2009,
                 'harvard_oxford':_load_harvard_oxford,
                 'pauli_2017':_load_pauli_2017,
                 'schaefer_2018': _load_schaefer_2018,
                 'yeo_2011': _load_yeo_2011,
                }

def parse_atlas(atlas_arg):
    atlas = None
    for key in ATLAS_ARGS.keys():
        if key == atlas_arg[:len(key)]:
            atlas = key
    assert atlas is not None, f'invalid atlas please choose from {AVAILABLE_ATLASES}'
    if atlas == atlas_arg:
        arg = None
    else:
        arg = atlas_arg.split(atlas+'_')[1]
    
    return atlas,arg
    
def get_atlas(atlas):
    assert atlas in AVAILABLE_ATLASES
    atlas, arg = parse_atlas(atlas)
    if arg is not None:
        return ATLAS_LOADER[atlas](arg)
    else:
        return ATLAS_LOADER[atlas]()


#AVAILABLE_ATLAS_LABELS = {atlas:list(get_atlas(atlas)[1].keys()) for atlas in AVAILABLE_ATLASES}
                     

def get_roi_mask(atlas, rois):
    assert len(rois) > 0
    atlas_map, atlas_label_mapping = get_atlas(atlas)
    atlas_map_data = atlas_map.get_fdata()
    atlas_roi_masked = [(atlas_map_data.astype(int)==int(atlas_label_mapping[roi])).astype(int) for roi in rois]
    atlas_roi_masked = np.array(atlas_roi_masked)
    atlas_roi_masked = atlas_roi_masked.sum(0)
    atlas_roi_masked = nib.Nifti1Image(atlas_roi_masked,affine=atlas_map.affine)
    return atlas_roi_masked

def get_roi_masks(atlas_map, atlas_label_mapping):
    arr = atlas_map.get_fdata()
    arr = arr.astype(int)
    roi_masks = {}
    for label, index in atlas_label_mapping.items():
        index = int(index)
        mask_arr = (arr==index) *1.0
        roi_masks[label]= nib.Nifti1Image(mask_arr,affine=atlas_map.affine)
        
    return roi_masks
    
    
def get_roi_mean_activation_from_atlas(nii_img, atlas):
    atlas_map, atlas_label_mapping = get_atlas(atlas)
    nii_img = resample_to_img(nii_img, atlas_map)
    atlas_map_data = atlas_map.get_fdata()
    nii_data = nii_img.get_fdata()
    mean_activation = {}
    for roi, index in atlas_label_mapping.items():
        mask = (atlas_map_data.astype(int)==int(index))
        roi_acts = nii_data[np.nonzero(mask)]
        mean_activation[roi] = roi_acts.mean()
    return mean_activation

def get_roi_mean_activation(nii_img, roi_masks):
    nii_img = resample_to_img(nii_img, roi_masks[list(roi_masks.keys())[0]])
    
    nii_data = nii_img.get_fdata()
    mean_activation = {}
    
    for roi, mask in roi_masks.items():
        mask = mask.get_fdata().astype(int)
        roi_acts = nii_data[np.nonzero(mask)]
        mean_activation[roi] = roi_acts.mean()
    return mean_activation

def get_roi_masked_img(nii_img, roi_masks):
    temp = roi_masks[list(roi_masks.keys())[0]]
    nii_img = resample_to_img(nii_img,temp)
    nii_data = nii_img.get_fdata()
    roi_masked = {}
    for roi, mask in roi_masks.items():
        mask = mask.get_fdata().astype(int)
        masked_img = nii_data[mask==1]
        roi_masked[roi] = nib.Nifti1Image(masked_img,affine=temp.affine)
    return roi_masked

def get_roi_masked_img_from_atlas(nii_img, atlas):
    atlas_map, atlas_label_mapping = get_atlas(atlas)
    roi_masks = get_roi_masks(atlas_map, atlas_label_mapping)
    temp = roi_masks[list(roi_masks.keys())[0]]
    nii_img = resample_to_img(nii_img,temp)
    nii_data = nii_img.get_fdata()
    roi_masked = {}
    for roi, mask in roi_masks.items():
        mask = mask.get_fdata().astype(int)
        masked_img = nii_data[mask==1]
        roi_masked[roi] = nib.Nifti1Image(masked_img,affine=temp.affine)
    return roi_masked