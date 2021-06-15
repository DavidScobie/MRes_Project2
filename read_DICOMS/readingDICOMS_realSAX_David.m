% function [allData, resampledImagesOut] = readingDICOMS_realSAX()

tot_folder = uigetdir;
tot_folder = [tot_folder]; % select the folder with the data starting '__2021'....'

folders = dir([tot_folder, '/__*'])

nSlices = size(folders, 1)

counter= 1;
for(s=1:2:nSlices) % we only take altrnate ones as the even ones are the ML result
    folder_name = [folders(s).folder, '/', folders(s).name]
    
    images_names = dir([folder_name, '/*.IMA']);
    nImages = size(images_names, 1)
    for(ph=1:nImages)
        Data(:,:,ph) = dicomread([images_names(ph).folder, '/', images_names(ph).name]);
    end
    
    allData{counter} = Data; % this is necessary as a struct as there are differentnumber of phases in each slice
    
    clear 'Data'
    counter = counter+1;
    
end


%%

% then you need some code to do the temporal interpolation and the
% normalisation of this data before you put it into h5v files and through
% the netowrk....something likethis:

nGriddedSlices = size(allData, 2)


for(s=1:nGriddedSlices)
    
    resampledImagesOut1= allData{s};
    nFrames = size(resampledImagesOut1, 3)

    [x,y,z] = meshgrid(1:192,1:192,1:nFrames);
    [x1,y1,z1] = meshgrid(1:192,1:192, 1:(nFrames-1)/19:nFrames);

    if(size(x1, 3) ~= 20)
        disp('ERROR');
    end

    norm_dat = interp3(x,y,z, double(resampledImagesOut1),x1,y1,z1);
    resampledImagesOut(:,:,:,s) = (norm_dat - min(norm_dat(:)))/(max(norm_dat(:)) - min(norm_dat(:)));
end

%% Saving the low resolution data
%DWS
 addpath 'C:/PHD/MRes_project/ML_work'
 
 low_res_data           = permute(resampledImagesOut,        [3 1 2 4]);
%  s = struct();
%  s.x = low_res_data;
no_subjects = size(resampledImagesOut);
 
 s = struct();
for(i=1:no_subjects(4)) 
    s(i).x = low_res_data(:,:,:,i);
end
 
save_dir = 'C:/PHD/MRes_project/ML_work/read_DICOMS/more_low_res_data';
dlexsave(save_dir, s, 'prefixes', 'd1');

%%
%Saving the ML reconstructions to check how good these look

counter= 1;
for(s=2:2:nSlices-1) % we only take altrnate ones as the even ones are the ML result
    folder_name = [folders(s).folder, '/', folders(s).name]
    
    images_names = dir([folder_name, '/*.IMA']);
    nImages = size(images_names, 1)
    for(ph=1:nImages)
        Data(:,:,ph) = dicomread([images_names(ph).folder, '/', images_names(ph).name]);
    end
    
    allData{counter} = Data; % this is necessary as a struct as there are differentnumber of phases in each slice
    
    clear 'Data'
    counter = counter+1;
    
end
%%
nGriddedSlices = size(allData, 2)

for(s1=1:nGriddedSlices - 1)
    
  slice = double(cell2mat(allData(s1)));

  ML_recon_out(:,:,:,s1) = double((slice - min(min(min(slice))))/(max(max(max(slice))) - min(min(min(slice)))));  

end

 addpath 'C:/PHD/MRes_project/ML_work'
 
 ML_recon           = permute((ML_recon_out),        [3 1 2 4]);
%  s = struct();
%  s.x = low_res_data;
 
 s = struct();
for(i=1:11) 
    s(i).y_pred = ML_recon(:,:,:,i);
end
 
save_dir = 'C:/PHD/MRes_project/ML_work/read_DICOMS/ML_recon';
% dlexsave(save_dir, s, 'prefixes', 'd1');

%%
% view a case from the ML reconstruction
seventh = allData{7};
implay(permute(seventh(32:160,32:160,:),[1 2 3]))
