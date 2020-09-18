clear all  % clear workspace

addpath('/usr/local/matlab_toolbox/spm12')

first_output_path = '/home/cheoljun/project_model_based_fmri/examples/fmri_analysis/results';
output_path = '/home/cheoljun/project_model_based_fmri/examples/fmri_analysis/results';

%% Initialise SPM defaults
spm('Defaults','fMRI');
spm_jobman('initcfg'); % SPM12 
%%

subjNum = 16;
condNum = 7;

allConds = {'gamble','safe','gamble_vs_safe','utility_gamble','utility_safe','utility_gamble_vs_safe','utility_total'}

for cond_i = 1:condNum
    
    cond = allConds{cond_i};  

    switch cond
        case 'gamble'
            contrast_num = '0001';
            dir_name = cond;
        case 'safe'
            contrast_num = '0002';
            dir_name = cond;
        case 'gamble_vs_safe'
            contrast_num = '0003';
            dir_name = cond;
        case 'utility_gamble'
            contrast_num = '0004';
            dir_name = cond;
        case 'utility_safe'
            contrast_num = '0005';
            dir_name = cond;
        case 'utility_gamble_vs_safe'
            contrast_num = '0006';
            dir_name = cond;
        case 'utility_total'
            contrast_num = '0007';
            dir_name = cond;
    end
    result_path = fullfile(output_path, cond);
    if isfolder(result_path)
        delete([result_path '/*']);
        rmdir(result_path);
    end
    
    mkdir(result_path);
    
    matlabbatch = [];
    
    scanFiles = [];
    for subj_n = 1:subjNum          
        con_file = fullfile(first_output_path, sprintf('sub-%02d',subj_n), ['con_' contrast_num '.nii,1'])
        scanFiles{subj_n,1} = con_file;
    end
    
    matlabbatch{1}.spm.stats.factorial_design.dir = {result_path};
    matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = scanFiles;
    
    matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
    
    spm_jobman('run', matlabbatch)
    disp(' model is specified');
    
    %% estimation
    matlabbatch = [];
    matlabbatch{1}.spm.stats.fmri_est.spmmat = { fullfile(result_path, 'SPM.mat') };
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    spm_jobman('run', matlabbatch)
    
    %% T-contrast (one-step t-test)
    matlabbatch = [];
    matlabbatch{1}.spm.stats.con.spmmat = { fullfile(result_path, 'SPM.mat') };
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = dir_name;
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = [1];
    matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{1}.spm.stats.con.delete = 1;
    spm_jobman('run', matlabbatch)
    
    disp( 'contrast is created')
    
end
