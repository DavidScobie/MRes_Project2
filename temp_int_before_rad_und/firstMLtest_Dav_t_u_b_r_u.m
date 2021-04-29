% Get truth data
% clc
% clear
% close all
addpath '/PHD/MRes_project/ML_work/paper_data_mat_files'
addpath '/PHD/MRes_project/ML_work'
addpath '/PHD/MRes_project/ML_work/gridder'
addpath 'D:/MRes_project/ML_work/paper_data_mat_files'
% load('D:\MRes_project\ML_work\paper_data_mat_files\SAXdataAll.mat');
% load('SAXdataAll.mat')
num_samples = length(new_dat_final)

% here i just forced it to only use the firstt 250
% if(num_samples > 250)  
%     num_samples = 250;
% end

%% 
% Either you can do your MRI experiemnt (in this case undersampling and
% gridding) in MATLAB or in dlex.
% if you do it in dlex just skip this whole secion...

% in MALTAB...

addpath '/Users/jennifer/Documents/work/Radial/DL'
%this is the same as from the 'runBatches3D_newRadialTraj' file which i sent you before
acc_fact = 13;

%  for(i=1:num_samples)
 for(i=1:5)
     
        %temporal interpolation
         newMatrix = 192;
        nFrames = size(new_dat_final{i}, 3)
        [x,y,z] = meshgrid(1:newMatrix,1:newMatrix,1:nFrames);
        [x1,y1,z1] = meshgrid(1:newMatrix,1:newMatrix, 1:(nFrames-1)/19:nFrames);
        disp([x1,y1,z1])
        if(size(x1, 3) ~= 20)
            disp('ERROR');
        end
        
        image_in = interp3(x,y,z, (new_dat_final{i}),x1,y1,z1);
        
        %radial undersampling truth to give undersampled
        simulated_sortGA = abs(SimulatingUndersampledRadialData_sortedGA_t_i_before_r_u(image_in, acc_fact));
        
        %Normalising both truth and undersampled
        [data_truth{i}, data_UnderSampled{i}] = resample_undersample_data_temp_int_before_rad_und(image_in, simulated_sortGA);
        
        disp('hi')
 end
 
% skip up to here is doing sampling in dlex
%%
%Load the data in
load('Test_Data_truth268.mat')
load('Test_Data_sortGA268.mat')

%Reduce the size down to 32x32x20
% images_truth=images_truth(49:80,49:80,1:20,:);
% images_sortGA=images_sortGA(49:80,49:80,1:20,:);

%Find how many images there are
image_dims=size(images_truth);
num_samples=image_dims(4);

%Reassign arrays
data_truth0=images_truth;
% data_Undersampled0=images_sortGA;

%Initialise empty cell arrays
data_truth=cell(1,num_samples);
% data_Undersampled=cell(1,num_samples);

%Turn the double arrays into cell arrays
for i = 1:num_samples
    data_truth{i} = mat2cell(data_truth0(:,:,:,i),image_dims(1),image_dims(2),image_dims(3));
%     data_Undersampled{i} = mat2cell(data_Undersampled0(:,:,:,i),image_dims(1),image_dims(2),image_dims(3));
end

% Permute to correct image dimensions

for(i=1:num_samples)
    data_truth{i}           = mat2cell(permute(cell2mat(data_truth{i}),        [3 1 2]),image_dims(3),image_dims(1),image_dims(2));
%     data_Undersampled{i}    = mat2cell(permute(cell2mat(data_Undersampled{i}), [3 1 2]),image_dims(3),image_dims(1),image_dims(2));
end

% Convert data to struct, and back to mat (from cell)
%
s = struct();
for(i=1:num_samples) 
    s(i).y = cell2mat(data_truth{i});
%     s(i).x = cell2mat(data_Undersampled{i});  
end
%%

% Use dlexsave to write to correct format into the ‘data’ folder

% save_dir = '/Users/jennifer/dlex/JAStest1/data/cine/'
save_dir = 'C:/PHD/MRes_project/ML_work/mapped_docker_files/ml/data/cine_denoising/'

dlexsave(save_dir, s, 'prefixes', 'd1')