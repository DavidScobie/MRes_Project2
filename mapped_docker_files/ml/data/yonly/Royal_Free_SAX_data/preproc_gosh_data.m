%This script takes in the Royal Free data. pads it to a square size, and
%interpolates across space and time to make it (192,192,40). Then the data
%is tiled out to 40 temporal frames, noise is added. Normalized and then saved as a
%structure

%Combine training and test data into 1 big dataset
% all_dat = [test_orig_data, train_orig_data];
all_rr_int_final = [test_rr_int_final, train_rr_int_final]; 

%Clear variables to save memory
clear test_orig_data
clear train_orig_data

s = struct();
%for i = 1:size(all_dat,2)
for i = 1989:2209 %Manually choose train and val split 1:1988 and 1989:2209
    %Read the data in
    dataIn = all_dat{i};
    
    %disp(i)
    
    %check if there are any nans in the data
    result = sum(isnan(dataIn(:)));
    if result > 0
        disp(i)
        disp(result)
        disp('there is at least 1 nan in the data before preproc')
    end
    
    %permute if necessary if the data is wrong way round
    if size(dataIn,1) == 222
        dataIn = permute(dataIn,[2 1 3]);
    elseif size(dataIn,1) == 224
        dataIn = permute(dataIn,[2 1 3]);
    end
    
    %Pad onto (240,240,nFrames)
    nFrames = size(dataIn,3);
    tempIn = zeros(240,240, nFrames);  
    %tempIn(pad(1)+1 : size(dataIn,1) + pad(1) , pad(2)+1 : size(dataIn,2)+pad(2), :) = dataIn;
    if size(dataIn,2) == 222
        tempIn(:,10:231,:) = dataIn(17:256, :, :);
    elseif size(dataIn,2) == 224
        tempIn(:,9:232,:) = dataIn(17:256, :, :);
    end
    
    %Interpolate to (192x192xnFrames) and to number of frames for 1 cycle with
    %temporal resolution of 36.4ms (rather than current resolution which is
    %similar but not 36.4. This can be found by dividing the rr-int by nFrames. 
    
    %No temporal interpolation if rr int is not given
    if all_rr_int_final(i) <= 1
        rr = nFrames .* 36.4;
    else
        rr = all_rr_int_final(i);
    end
    
    %Interpolate in space and time
    time = rr/nFrames;
    time_dat = 0:time:rr-time;
    [x,y,z] = meshgrid(0:239, 0:239, time_dat);
    newSp = 240/192;
    nPos = floor(rr/36.4);
    [x1,y1,z1] = meshgrid(0:newSp:239, 0:newSp:239, 0:36.4:(nPos-1)*36.4);
    Interped = interp3(x,y,z, double(tempIn),x1,y1,z1);
    
    %If the last frame is full of nans then get rid of last frame
    if sum(isnan(Interped(:))) > 0
        Interped = Interped(:,:,1:nPos - 1);
        time_repeated = zeros(192,192,(nPos-1)*4);
    else
        time_repeated = zeros(192,192,nPos*4);
    end
    
    %Repeat the data out to 40 frames
    time_40_fram = zeros(192,192,40);
    time_repeated(:,:,:) = repmat(Interped,[1 1 4]);
    time_40_fram(:,:,:) = time_repeated(:,:,1:40);
    
    %Add some noise to the data
    avData = mean(time_40_fram, 3);    
    maxVal = max(avData(:));
    k=find(avData < maxVal*0.2);
    temp = avData(k);
    meanVal = mean(temp);
    k=find(time_40_fram < meanVal);
    meanNoise = mean(time_40_fram(k));
    k=find(time_40_fram < meanNoise);
    time_40_fram(k) = meanNoise + rand(size(k))*meanNoise;
    
    %Normalize
    if max(max(max(time_40_fram))) > 0 %dont want to create nans 
        time_40_fram(:,:,:) = time_40_fram(:,:,:) ./ max(max(max(time_40_fram)));
    end
    
    %Permute to correct image dimensions
    time_40_fram_perm(:,:,:) = permute(time_40_fram(:,:,:),[3 1 2]);
    
    %check if there are any nans in the data
    result = sum(isnan(time_40_fram_perm(:)));
    if result > 0
        disp(i)
        disp(result)
        disp('there is at least 1 nan in the data after preproc')
        
        %disp(find(isnan(time_40_fram_perm))); %Finding indicies of Nans
    end
    
    %Check if the max value of any of the data is 0
    if max(time_40_fram_perm(:)) < 1
        disp(i)
        disp('this dataset contains all zeros')
    end
    
    %Save it as a structure
    s(i-1988).y = time_40_fram_perm;
end

save_dir = 'C:/PHD/MRes_project/ML_work/read_DICOMS/data/Royal_Free_SAX_data/RF_full_set_2/';

dlexsave(save_dir, s, 'prefixes', 'val');
