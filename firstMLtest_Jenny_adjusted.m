% Get truth data
%clc
%clear
%close all

load('SAXdataAll.mat')
num_samples = length(new_dat_final);

% here i just forced it to only use the first 40
if(num_samples > 40)  
    num_samples = 40;
end

bIfPreProcessingInMATALB = false;


%% 
% Either you can do your MRI experiemnt (in this case undersampling and
% gridding) in MATLAB or in dlex.
% if you do it in dlex just skip this whole secion...

if(bIfPreProcessingInMATALB == true)
    % in MALTAB...

%     addpath '/Users/jennifer/Documents/work/Radial/DL'
    addpath 'C:/PHD/MRes_project/ML_work/gridder'
    %this is the same as from the 'runBatches3D_newRadialTraj' file which i sent you before
    acc_fact = 13;

     for(i=1:num_samples)
            simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA(new_dat_final{i}, acc_fact));

            [data_truth{i}, data_UnderSampled{i}] = resample_undersample_data(new_dat_final{i}, simulated_sortGA);
     end

    % skip up to here is doing sampling in dlex
    
else
    for(i=1:num_samples)
    	data_truth{i} = new_dat_final{i};
     end
end
% Permute to correct image dimensions

for(i=1:num_samples)
    data_truth{i}           = permute(data_truth{i},        [3 1 2]);
    if(bIfPreProcessingInMATALB == true)
        data_UnderSampled{i}    = permute(data_UnderSampled{i}, [3 1 2]);
    end
end

% Convert data to struct

s = struct();
for(i=1:num_samples) 
    s(i).y = data_truth{i};
    if(bIfPreProcessingInMATALB == true)
        s(i).x = data_UnderSampled{i};
    end
end

% Use dlexsave to write to correct format into the ‘data’ folder

save_dir = 'C:/PHD/MRes_project/ML_work/mapped_docker_files/ml/data/training_192siz_MAT_preproc/'

dlexsave(save_dir, s, 'prefixes', 'd1');