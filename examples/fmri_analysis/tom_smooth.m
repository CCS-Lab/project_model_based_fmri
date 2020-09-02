% Smoothing of preprocessed Tom 2007 data

%2020.09.01 Cheol Jun Cho

addpath('/usr/local/matlab_toolbox/spm12')

smoothing = [8 8 8];  % smoothing 

subjNum = 16;
runNum = 3;


spm('defaults', 'FMRI');
spm_jobman('initcfg'); % SPM12
    
for subj_n = 1:subjNum
    
    for run_n = 1:runNum
        matlabbatch = [];
        subj_id =  sprintf('sub-%02d',subj_n);
        tmpFile = fullfile(fmri_path, subj_id, 'func', [subj_id '_' taskName '_' sprintf('run-%d',run_n) '_' 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']);
        gunzip(tmpFile);
        tmpFile = fullfile(fmri_path, subj_id, 'func', [subj_id '_' taskName '_' sprintf('run-%d',run_n) '_' 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii']);
        
        % get header information to read a 4D file
        tmpHdr = spm_vol(tmpFile);
        f_list_length = size(tmpHdr, 1);  % number of 3d volumes
        for jx = 1:f_list_length
            scanFiles{jx,1} = [tmpFile ',' num2str(jx)] ; % add numbers in the end
            % End of difference for 3D vs. 4D %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        matlabbatch{1}.spm.spatial.smooth.data = scanFiles;
        matlabbatch{1}.spm.spatial.smooth.fwhm = smoothing;
        matlabbatch{1}.spm.spatial.smooth.dtype = 0;
        matlabbatch{1}.spm.spatial.smooth.im = 0;
        matlabbatch{1}.spm.spatial.smooth.prefix = 'smoothed_';
        spm_jobman('run', matlabbatch) 
    end
end