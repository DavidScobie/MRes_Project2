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

% here i just forced it to only use the first 40
% if(num_samples > 40)  
%     num_samples = 40;
% end

bIfPreProcessingInMATALB = false;


%% 
% Either you can do your MRI experiemnt (in this case undersampling and
% gridding) in MATLAB or in dlex.
% if you do it in dlex just skip this whole secion...

if(bIfPreProcessingInMATALB == true)
    % in MALTAB...

%     addpath '/Users/jennifer/Documents/work/Radial/DL'
    %this is the same as from the 'runBatches3D_newRadialTraj' file which i sent you before
    acc_fact = 13;
    
no_frames_collect = 40;
temp_res = 36.4;
scan_time = no_frames_collect .* temp_res;
    
for(i=1:num_samples)

    fram_40_data = make_40_frame_data(rr_int(i), scan_time, new_dat_final{i});
    
    %randi for start of cardiac cycle
    %resample dataset based on rr interval and no frames going to collect
    %create new dataset based on newdatfinal which now has 40 timepoints always


            simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA(fram_40_data, acc_fact));
            
%             If you want 192 size matrix use this
            [data_truth{i}, data_UnderSampled{i}] = frame_40_res_und_data(fram_40_data, simulated_sortGA);
            disp(i)
     end

    % skip up to here is doing sampling in dlex
    %%
else
    for(i=1:num_samples)
    	data_truth{i} = new_dat_final{i};
     end
end
% Permute to correct image dimensions

for(i=1:num_samples)
%if doing last 268   
% for (i=1:268) 
    data_truth{i}           = permute(data_truth{i},        [3 1 2]);
%     if(bIfPreProcessingInMATALB == true)
    data_UnderSampled{i}    = permute(data_UnderSampled{i}, [3 1 2]);
%     end
end

% Convert data to struct

s = struct();
for(i=1:num_samples) 

%If doing last 268
% for(i=1:268)
    s(i).y = data_truth{i};
%     if(bIfPreProcessingInMATALB == true)
    s(i).x = data_UnderSampled{i};
%     end
end

% Use dlexsave to write to correct format into the ‘data’ folder

save_dir = 'C:/PHD/MRes_project/ML_work/mapped_docker_files/ml/data/frames_40/'

dlexsave(save_dir, s, 'prefixes', 'test');