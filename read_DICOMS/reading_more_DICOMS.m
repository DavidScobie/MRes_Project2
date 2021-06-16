%Read the data in from the file you select
tot_folder = uigetdir;
tot_folder = [tot_folder];

all_filenames = dir([tot_folder,'/*.ima']);
no_files = length(all_filenames);

unique_slice_no = strings; %find the unique slice numbers
for i = 1:no_files
    %Use the positions of the dots in the filename to find the slice numbers
    filename = all_filenames(i);
    dot_indics = strfind(filename.name,'.'); 
    dot_2_pos = dot_indics(2);
    dot_3_pos = dot_indics(3);
    slice_string = filename.name(dot_2_pos+1:dot_3_pos-1);
    
    %Make an array of all the slice numbers
    if any(unique_slice_no(:) == slice_string)
        % do nothing
    else
        unique_slice_no(length(unique_slice_no)+1) = slice_string;
    end

end

%The start of the filename is unique to the patient
filename = all_filenames(1);
dot_indics = strfind(filename.name,'.');
dot_2_pos = dot_indics(2);
dot_3_pos = dot_indics(3);
patient_string = filename.name(1:dot_2_pos);

%Read the DICOM images in to make matrices of the data
for i = 2:length(unique_slice_no)
    test=dir([tot_folder,'\',patient_string,char(unique_slice_no(i)),'*.IMA']);
    for j = 1:size(test)
        dico{i-1}(:,:,j) = dicomread([test(j).folder, '/', test(j).name]);
    end
end

%The gridded data are the odd indicies
for i = 1:2:length(dico)
    gridded{(i+1)./2} = dico{i};
end

%The MLrecon data are the even indicies
for i = 2:2:length(dico)
    ML_recon{i./2} = dico{i};
end

%%

% then you need some code to do the temporal interpolation and the
% normalisation of this data before you put it into h5v files and through
% the netowrk....something likethis:

for i = 1:size(gridded,2)
 
    resampledImagesOut1= gridded{i};
    nFrames = size(resampledImagesOut1, 3);

    [x,y,z] = meshgrid(1:192,1:192,1:nFrames);
    [x1,y1,z1] = meshgrid(1:192,1:192, 1:(nFrames-1)/19:nFrames);

    if(size(x1, 3) ~= 20)
        disp('ERROR');
    end

    norm_dat = interp3(x,y,z, double(resampledImagesOut1),x1,y1,z1);
    resampledImagesOut(:,:,:,i) = (norm_dat - min(norm_dat(:)))/(max(norm_dat(:)) - min(norm_dat(:)));
end

%% Saving the low resolution data
%DWS
addpath 'C:/PHD/MRes_project/ML_work'
 
gridded_data = permute(resampledImagesOut,        [3 1 2 4]);
 
s = struct();
for i=1:size(resampledImagesOut,4) 
    s(i).x = gridded_data(:,:,:,i);
end

save_dir = 'C:/PHD/MRes_project/ML_work/read_DICOMS/data/scanner_recon/pat_13_SAX';
% dlexsave(save_dir, s, 'prefixes', 'd1');
%% Saving the scanner reconstructions
s = struct();
for i=1:size(ML_recon,2) 
    s(i).y_pred = ML_recon{i};
end
 
save_dir = 'C:/PHD/MRes_project/ML_work/read_DICOMS/scanner_reconstruction/pat_5';
dlexsave(save_dir, s, 'prefixes', 'd1');
