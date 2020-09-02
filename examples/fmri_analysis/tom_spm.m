% First-level analysis of Mixed-gamble task from Tom 2007
% Modified by Cheol Jun Cho based on cluster_fnirs_PRL_1st_hy_v5.m by Hoyoung Doh
%
% download and add "tsvread.m" from https://kr.mathworks.com/matlabcentral/fileexchange/32782-tsvread-importing-tab-separated-data?focused=a53e9d7b-eac4-4992-21fa-d380115d33e5&tab=function
% run this code under each subject's parent folders

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% general specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all  % clear workspace

addpath('/usr/local/matlab_toolbox/spm12')

TR = 2;  % TR of the fMRI data
disp('Lets rock');
%% set ID, def_patz
defThres = 0.8;   % default threshold (increase it 0.8) thresholding voxel (mean actiation of each voxel should be greater than 20% of global mean) 
currApproach = 'tom_spm';  % current approach.. 
subjNum = 16;
runNum = 3;

taskName = 'task-mixedgamblestask';
motionregName = 'movement_regressors_tom';


% output_path - where to save output
output_path = '/home/cheoljun/project_model_based_fmri/examples/fmri_analysis/results';
fmri_path = '/home/cheoljun/project_model_based_fmri/examples/output/fmriprep';
behav_root = '/home/cheoljun/project_model_based_fmri/examples/data/tom_2007/ds000005/';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% spm specification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spm('defaults', 'FMRI');
spm_jobman('initcfg'); % SPM12

matlabbatch = [];

disp('start spm specification')

for subj_n = 1:subjNum
    subj_id =  sprintf('sub-%02d',subj_n);
    indiv_result_path = fullfile(output_path,subj_id)
    
    if isfolder(indiv_result_path)
        delete([indiv_result_path '/*'])
        rmdir(indiv_result_path)
    end
    
    mkdir(indiv_result_path)
    
    
    matlabbatch{subj_n}.spm.stats.fmri_spec.dir = {indiv_result_path };
    matlabbatch{subj_n}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{subj_n}.spm.stats.fmri_spec.timing.RT = TR;
    matlabbatch{subj_n}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{subj_n}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
    
    
    for run_n = 1:runNum
        
        move_path_origin = fullfile(fmri_path, subj_id, 'func', [subj_id '_' taskName '_' sprintf('run-%d',run_n) '_' 'desc-confounds_regressors.tsv']);
        [data, header, ] = tsvread(move_path_origin);
        trans_x = strmatch('trans_x', header, 'exact');
        trans_y = strmatch('trans_y', header, 'exact');
        trans_z = strmatch('trans_z', header, 'exact');
        rot_x = strmatch('rot_x', header, 'exact');
        rot_y = strmatch('rot_y', header, 'exact');
        rot_z = strmatch('rot_z', header, 'exact');
        R_mov = data(2:end, [trans_x,trans_y,trans_z,rot_x,rot_y,rot_z]);  % remove the first row, 26-31 columns --> movement regressors
        R_mov = fillmissing(R_mov, 'nearest');
        R_mov(~isfinite(R_mov))=0;
        outliers = startsWith(header,'motion_outlier');
        R_outlier = data(2:end, [outliers]);
        R = horzcat(R_mov, R_outlier);
        motionreg_save_path = [output_path '/' subj_id '/' motionregName '_' sprintf('run-%02d',run_n) '.mat'];
        save (motionreg_save_path, 'R');
    
        
        behav_path = fullfile(behav_root, subj_id, 'func', [subj_id '_' taskName '_' sprintf('run-%02d',run_n) '_' 'events.tsv']);
        
        behav_data = tdfread(behav_path,'\t')
        onset = behav_data.onset;
        duration = behav_data.duration;
        respnum = behav_data.respnum;


        % rescan files
        tmpFile = fullfile(fmri_path, subj_id, 'func', ['smoothed_' subj_id '_' taskName '_' sprintf('run-%d',run_n) '_' 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii']);  
        
        % get header information to read a 4D file
        tmpHdr = spm_vol(tmpFile);
        f_list_length = size(tmpHdr, 1);  % number of 3d volumes
        for jx = 1:f_list_length
            scanFiles{jx,1} = [tmpFile ',' num2str(jx) ]; % add numbers in the end
            % End of difference for 3D vs. 4D %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end

        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).scans = scanFiles;

        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).name = 'gamble';
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).onset = onset(respnum <= 2);
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).duration = duration(respnum <= 2);
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).tmod = 0; 
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {}); % ?
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(1).orth = 0;

        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).name = 'safe';
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).onset = onset(respnum > 2);
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).duration = duration(respnum > 2);
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).tmod = 0; 
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {}); % ?
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).cond(2).orth = 0;

        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).multi = {''};
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).regress = struct('name', {}, 'val', {});
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).multi_reg = {fullfile(motionreg_save_path)};
        matlabbatch{subj_n}.spm.stats.fmri_spec.sess(run_n).hpf = 128;

    end
    
    matlabbatch{subj_n}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{subj_n}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{subj_n}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{subj_n}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{subj_n}.spm.stats.fmri_spec.mthresh = defThres;   % threshold
    matlabbatch{subj_n}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{subj_n}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
end

spm_jobman('run', matlabbatch) 
disp('parametric model is specified for all subjects')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% spm estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matlabbatch = [];

disp('start model estimation')

for subj_n = 1:subjNum
    subj_id =  sprintf('sub-%02d',subj_n);
    indiv_result_path = fullfile(output_path,subj_id)
       
    matlabbatch{subj_n}.spm.stats.fmri_est.spmmat = { fullfile( indiv_result_path, 'SPM.mat' ) };
    matlabbatch{subj_n}.spm.stats.fmri_est.method.Classical = 1;
    
end

spm_jobman('run', matlabbatch) 
disp('parametric model is estimated')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% spm contrast
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matlabbatch = [];

disp('start contrast generation')

for subj_n = 1:subjNum
    subj_id =  sprintf('sub-%02d',subj_n);
    indiv_result_path = fullfile(output_path,subj_id)
    
    matlabbatch{subj_n}.spm.stats.con.spmmat = { fullfile( indiv_result_path, 'SPM.mat'  )  };

    matlabbatch{subj_n}.spm.stats.con.consess{1}.tcon.name = 'gamble';
    matlabbatch{subj_n}.spm.stats.con.consess{1}.tcon.convec = [1/2];
    matlabbatch{subj_n}.spm.stats.con.consess{1}.tcon.sessrep = 'repl'; 

    matlabbatch{subj_n}.spm.stats.con.consess{2}.tcon.name = 'safe';
    matlabbatch{subj_n}.spm.stats.con.consess{2}.tcon.convec = [0 1/2];
    matlabbatch{subj_n}.spm.stats.con.consess{2}.tcon.sessrep = 'repl'; 
    
    matlabbatch{subj_n}.spm.stats.con.consess{3}.tcon.name = 'gamble_vs_safe';
    matlabbatch{subj_n}.spm.stats.con.consess{3}.tcon.convec = [1/2 -1/2];
    matlabbatch{subj_n}.spm.stats.con.consess{3}.tcon.sessrep = 'repl'; 

    matlabbatch{subj_n}.spm.stats.con.delete = 0; % after creating all contrasts

end

spm_jobman('run', matlabbatch) 
disp('contrasts are generated')