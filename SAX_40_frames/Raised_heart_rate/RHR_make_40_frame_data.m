function [inter_40_data] = RHR_make_40_frame_data(rrint, scan_time, input_data)
%You feed in the rr-interval, scan time and data.
%This function takes any number of temporal frames and repeats the data to
%make 40 frames. Starting at a random cardiac phase.

% accel = 1.8;
accel = normrnd(2,0.5); %Make a random acceleration factor
RHR_RR = rrint/accel; %Find new decreased rr interval

first_frame_no = randi(floor(RHR_RR)); %Find the random cardiac phase
full_seq_length = first_frame_no + scan_time; 
n_c_cycles = ceil(full_seq_length ./ RHR_RR); %Find number of cardiac cycles
long_data = repmat(input_data,[1 1 n_c_cycles+1]); %Extend the data out
first_slice = ceil((first_frame_no./RHR_RR).*(size(input_data,3))); %Find random first slice
% fram_40_data = long_data(:,:,first_slice:first_slice+39);
long_ran_start = long_data(:,:,first_slice:size(long_data,3)); %Long dataset with random first slice

% nFrames = size(truthImagesIn, 3)
% [x,y,z] = meshgrid(1:newMatrix,1:newMatrix,1:nFrames);
% [x1,y1,z1] = meshgrid(1:newMatrix,1:newMatrix, 1:(nFrames-1)/19:nFrames);
% truthImagesOut1 = double(truthImagesIn(startY:endY,startX:endX,:));
% norm_dat = interp3(x,y,z, (truthImagesOut1),x1,y1,z1);

nFrames = size(long_ran_start, 3); %Interpolate based on acceleration factor
[x,y,z] = meshgrid(1:192,1:192,1:nFrames);
[x1,y1,z1] = meshgrid(1:192, 1:192, 1:((nFrames-1)./((nFrames./accel)-1)):nFrames);
inter_data = interp3(x,y,z, (long_ran_start),x1,y1,z1);

inter_40_data = inter_data(:,:,1:40); %Chop interpolated data to 40 frames
end

