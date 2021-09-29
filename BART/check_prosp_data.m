%load in the slices of the prospective gridded data nd flip them the right way round 
load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_0_1.mat')
flipped_img_0_1 = flip(img_data,3);

load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_2_3.mat')
flipped_img_2_3 = flip(img_data,3);

load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_4_5.mat')
flipped_img_4_5 = flip(img_data,3);

load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_6_7.mat')
flipped_img_6_7 = flip(img_data,3);

load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_8_9.mat')
flipped_img_8_9 = flip(img_data,3);

load('C:\PHD\MRes_project\ML_work\BART\data\meas_ex3_stack\gridded_image_data\meas_MID00576_FID48987_ex3_stack_sl_10_11.mat')
flipped_img_10_11 = flip(img_data,3);

%Stack all of the slices together into 1 data object
img_0123 = cat(1,flipped_img_0_1,flipped_img_2_3);
img_012345 = cat(1,img_0123,flipped_img_4_5);
img_01234567 = cat(1,img_012345,flipped_img_6_7);
img_0123456789 = cat(1,img_01234567,flipped_img_8_9);
full_img = cat(1,img_0123456789,flipped_img_10_11);

normed_img = full_img./ max(max(max(max(full_img)))); %Normalise

slice1 = abs(squeeze(normed_img(7,:,:,:)));

implay(double(permute(slice1,[2 3 1])))

%Combine data into a structure and save as h5 files
dims = size(normed_img);
s = struct();
for i = 1:dims(1)
    s(i).x = abs(squeeze(normed_img(i,:,:,:)));
end

save_dir = 'C:/PHD/MRes_project/ML_work/BART/data/meas_ex3_stack/h5_slices';
% dlexsave(save_dir, s, 'prefixes', 'd1');

%%

%Just looking at 1 slice and saving it

load('C:\PHD\MRes_project\ML_work\BART\data\meas_rest_stack\gridded_image_data\meas_MID00573_FID48984_rest_stack_sl_0.mat')
flipped_img_n = flip(img_data,3);

normed_img = flipped_img_n./ max(max(max(max(flipped_img_n)))); %Normalise

slice1 = abs(squeeze(normed_img(1,:,:,:)));

implay(double(permute(slice1,[2 3 1])))

dims = size(normed_img);
s = struct();
for i = 1:dims(1)
    s(i).x = abs(squeeze(normed_img(i,:,:,:)));
end

save_dir = 'C:/PHD/MRes_project/ML_work/BART/data/meas_rest_stack/h5_slices';
% dlexsave(save_dir, s, 'prefixes', 'd1');