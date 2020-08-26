#!/bin/bash

################################################################################
# Global settings
################################################################################

FMRIPREP_VER=20.1.0
SIMG_FILE=/data/singularity/images/fmriprep-${FMRIPREP_VER}.simg

BIDS_DIR='/home/cheoljun/project_model_based_fmri/examples/data/tom_2007/ds000005'
OUTPUT_DIR='/home/cheoljun/project_model_based_fmri/examples/output'
WORK_DIR='/home/cheoljun/project_model_based_fmri/examples'

FS_LICENSE='/home/cheoljun/project_model_based_fmri/examples/license.txt'

################################################################################
# Run singularity image for fmriprep
################################################################################

# 1) Build a fmriprep image using singularity from docker hub
#
# USAGE: singularity build [image-file] [source]
#
if [ ! -f "$SIMG_FILE" ]; then
	singularity build\ 
		$SIMG_FILE \ 
		docker://poldracklab/fmriprep:${FMRIPREP_VER}
fi

# 2) Run the built fmriprep image
#
singularity run \
	-B $BIDS_DIR:/data \
	-B $OUTPUT_DIR:/out \
	-B $WORK_DIR:/work \
	--cleanenv $SIMG_FILE \
	/data \
	/out \
	participant \
	--fs-license-file $FS_LICENSE \
	--nthreads 16 \
	--mem_mb 16000 \
	--work-dir /work \
	--fs-no-reconall \
			 
