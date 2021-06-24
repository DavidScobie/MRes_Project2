% Get truth data
%clc
%clear
%close all
addpath 'D:\MRes_project\ML_work\paper_data_mat_files'
addpath 'C:\PHD\MRes_project\ML_work\gridder' 
% addpath '/Users/jennifer/Documents/work/Radial/DL'
load('SAXdataAll.mat');
num_samples = 2000;

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


     for(i=1:num_samples)

    %If you just want to do the last 268 sets use
%      for(i=2001:2268)


            simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA(new_dat_final{i}, acc_fact));
            
            %If you want 192 size matrix use this
%             [data_truth{i}, data_UnderSampled{i}] = resamp_undersamp_dat_192(new_dat_final{i}, simulated_sortGA);
            
            %If you just want to do the last 268 sets use
%               [data_truth{i-2000}, data_UnderSampled{i-2000}] = resamp_undersamp_dat_192(new_dat_final{i}, simulated_sortGA);
            
            %If you want 128 size matrix (cropped) use this
            [data_truth{i}, data_UnderSampled{i}] = resample_undersample_data(new_dat_final{i}, simulated_sortGA);
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

save_dir = 'C:/PHD/MRes_project/ML_work/mapped_docker_files/ml/data/last_268_192_MAT_pre/'

dlexsave(save_dir, s, 'prefixes', 'test');

%%
third_dim = [];
for i = 1:2268
    third_dim(i) = size(new_dat_final{i},3);
end