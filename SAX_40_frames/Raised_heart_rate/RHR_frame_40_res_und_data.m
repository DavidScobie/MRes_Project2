function [truthImagesOut, resampledImagesOut1] = RHR_frame_40_res_und_data(truthImagesIn, resampledImagesIn1, accel)
% This function takes in the truth, undersampled data and the heart rate acceleration factor. The data will be
%long and beginning at a random cardiac phase. 
%This function uses the accleration factor to temporally interpolate the 
%frames (effectively speeding up the heart rate).
%It then only takes the first 40 frames of this for truth and undersampled
%inputs.
%We then normalise and output.


%% Truth images
nFrames = size(truthImagesIn, 3);
[x,y,z] = meshgrid(1:192,1:192,1:nFrames);
[x1,y1,z1] = meshgrid(1:192, 1:192, 1:((nFrames-1)./((nFrames./accel)-1)):nFrames);
inter_data = interp3(x,y,z, (truthImagesIn),x1,y1,z1);
truth_40_data = inter_data(:,:,1:40); %Chop interpolated data to 40 frames

norm_dat = truth_40_data;

min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));

truthImagesOut = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
truthImagesOut = cast(truthImagesOut, 'single');

%% Undersampled images

norm_dat = interp3(x,y,z, (resampledImagesIn1),x1,y1,z1);
norm_dat = norm_dat(:,:,1:40);
min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));
resampledImagesOut1 = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
resampledImagesOut1 = cast(resampledImagesOut1, 'single');


return;
