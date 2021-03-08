clear all
close all

addpath C:\PHD\MRes_project\ML_work\dlex_framework\dlex\matlab
addpath C:\PHD\MRes_project\ML_work\gridder

%Loading data in
val_data=dlexload('C:\PHD\MRes_project\ML_work\mapped_docker_files\ml\results\same_as_paper\e_150_192_siz_in_and_out_bs4\pred\val');
%%
%Y IMAGES
for i =1:200    
    % Getting truth image from read in data
    truthImagesIn = val_data(i).y;
    %Permuting so that dimensions work with resample_undersample_data
    truthImagesIn = permute(truthImagesIn,[2 3 1]);

    newMatrix = 128;

    origMatrix = size(truthImagesIn, 1);

    r_d_fft = itok(truthImagesIn,3);
    mid = ceil(size(r_d_fft, 3) / 2);
    r_d_fft(:,:,mid-1:mid+1) = 0;             % what is this for??
    moving_heart = abs(sum(r_d_fft,3));

    m_h_x = sum(moving_heart,1); 
    m_h_y = sum(moving_heart,2);

    com_x = (sum(m_h_x .* [1:length(m_h_x)]))/sum(m_h_x);
    com_y = (sum(m_h_y .* [1:length(m_h_y)]'))/sum(m_h_y);

    addY = 0;
    startY = round(com_y)-(newMatrix/2);
    if(startY < 1) 
        addY = -startY + 1;
        startY = 1;
    end;
    endY = addY + round(com_y)+(newMatrix/2 - 1);
    if(endY > origMatrix)
        endY = origMatrix;
        startY = (newMatrix/2 +1);
    end

    addX = 0;
    startX = round(com_x)-(newMatrix/2);
    if(startX < 1) 
        addX = -startX + 1;
        startX = 1;
    end;
    endX = addX + round(com_x)+(newMatrix/2 - 1);
    if(endX > origMatrix)
        endX = origMatrix;
        startX = newMatrix/2 +1;
    end

    truthImagesOut1 = double(truthImagesIn(startY:endY,startX:endX,:));


    %% Y_PRED IMAGES
    % Getting pred image from read in data
    predImagesIn = val_data(i).y_pred;

    %Permuting so that dimensions work with resample_undersample_data
    predImagesIn = permute(predImagesIn,[2 3 1]);

    predImagesOut1 = double(predImagesIn(startY:endY,startX:endX,:));

    %% SSIM
    ssimval(i) = ssim(predImagesOut1,truthImagesOut1);
end
avg_ssim = (sum(ssimval))./size(ssimval,2);