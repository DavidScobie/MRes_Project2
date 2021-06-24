% Get truth data
%clc
%clear
%close all
addpath 'D:\MRes_project\ML_work\paper_data_mat_files'
addpath 'C:\PHD\MRes_project\ML_work\gridder' 
addpath 'C:\PHD\MRes_project\ML_work'

load('SAXdataAll.mat');
% num_samples = 2000;
num_samples = size(new_dat_final,2);

%% 

acc_fact = 13;  
no_frames_collect = 40;
temp_res = 36.4;
scan_time = no_frames_collect .* temp_res;
    
for(i=1:num_samples)
% for(i=1:2268)
    
    [long_ran_start, accel] = RHR_long_random_start(rr_int(i), scan_time, new_dat_final{i});
    
    %randi for start of cardiac cycle
    %resample dataset based on rr interval and no frames going to collect
    %create new dataset based on newdatfinal which now has 40 timepoints always


            simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA(long_ran_start, acc_fact));
            
%             If you want 192 size matrix use this
            [data_truth{i}, data_UnderSampled{i}] = RHR_frame_40_res_und_data(long_ran_start, simulated_sortGA, accel);
            disp(i)
end

% skip up to here is doing sampling in dlex
%%
% Permute to correct image dimensions

% for(i=1:num_samples)  
for (i=1:268) 
    data_truth{i}           = permute(data_truth{i},        [3 1 2]);
    data_UnderSampled{i}    = permute(data_UnderSampled{i}, [3 1 2]);
end

% Convert data to struct

s = struct();
% for(i=1:num_samples) 
for(i=1:268) 
    s(i).y = data_truth{i};
    s(i).x = data_UnderSampled{i};
end

% Use dlexsave to write to correct format into the ‘data’ folder

save_dir = 'C:/PHD/MRes_project/ML_work/mapped_docker_files/ml/data/frames_40_val/'

dlexsave(save_dir, s, 'prefixes', 'val');