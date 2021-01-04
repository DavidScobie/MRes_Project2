% Get truth data
clc
clear
close all

load('SAXdataAll.mat')
num_samples = length(new_dat_final)

% here i just forced it to only use the firstt 250
if(num_samples > 250)  
    num_samples = 250;
end

%% 
% Either you can do your MRI experiemnt (in this case undersampling and
% gridding) in MATLAB or in dlex.
% if you do it in dlex just skip this whole secion...

% in MALTAB...

addpath '/Users/jennifer/Documents/work/Radial/DL'
%this is the same as from the 'runBatches3D_newRadialTraj' file which i sent you before
acc_fact = 13;

 for(i=1:num_samples)
        simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA(new_dat_final{i}, acc_fact));
             
        [data_truth{i}, data_UnderSampled{i}] = resample_undersample_data(new_dat_final{i}, simulated_sortGA);
 end
 
% skip up to here is doing sampling in dlex
%%
%Load the data in
load('TestData_truth_268.mat')
load('TestData_sortGA_268.mat')

%Find how many images there are
image_dims=size(images_truth);
num_samples=image_dims(4);

%Reassign arrays
data_truth0=images_truth;
data_Undersampled0=images_sortGA;

%Initialise empty cell arrays
data_truth=cell(1,268);
data_Undersampled=cell(1,268);

%Turn the double arrays into cell arrays
for i = 1:num_samples
    data_truth{i} = mat2cell(data_truth0(:,:,:,i),image_dims(1),image_dims(2),image_dims(3));
    data_Undersampled{i} = mat2cell(data_Undersampled0(:,:,:,i),image_dims(1),image_dims(2),image_dims(3));
end

% Permute to correct image dimensions

for(i=1:num_samples)
    data_truth{i}           = mat2cell(permute(cell2mat(data_truth{i}),        [3 1 2]),image_dims(3),image_dims(1),image_dims(2));
    data_Undersampled{i}    = mat2cell(permute(cell2mat(data_Undersampled{i}), [3 1 2]),image_dims(3),image_dims(1),image_dims(2));
end

% Convert data to struct

s = struct();
for(i=1:num_samples) 
    s(i).y = data_truth{i};
    s(i).x = data_Undersampled{i};  
end


% Use dlexsave to write to correct format into the ‘data’ folder

save_dir = '/Users/jennifer/dlex/JAStest1/data/cine/'

dlexsave(save_dir, s, 'prefixes', 'dataSet1_');