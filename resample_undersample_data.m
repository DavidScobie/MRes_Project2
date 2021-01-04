function [truthImagesOut, resampledImagesOut1, resampledImagesOut2, resampledImagesOut3, resampledImagesOut4] = resample_undersample_data(truthImagesIn, resampledImagesIn1, resampledImagesIn2, resampledImagesIn3, resampledImagesIn4)

sizeIn = nargin;
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


nFrames = size(truthImagesIn, 3)
[x,y,z] = meshgrid(1:newMatrix,1:newMatrix,1:nFrames);
[x1,y1,z1] = meshgrid(1:newMatrix,1:newMatrix, 1:(nFrames-1)/19:nFrames);

if(size(x1, 3) ~= 20)
    disp('ERROR');
end


%% JAS start
truthImagesOut1 = double(truthImagesIn(startY:endY,startX:endX,:));
norm_dat = interp3(x,y,z, (truthImagesOut1),x1,y1,z1);

min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));

truthImagesOut = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
truthImagesOut = cast(truthImagesOut, 'single');

%% JAS end
resampledImagesOut1a = double(resampledImagesIn1(startY:endY,startX:endX,:));
%% JAS start
%norm_dat = interpft(resampledImagesOut1a, 20, 3);
norm_dat = interp3(x,y,z, (resampledImagesOut1a),x1,y1,z1);
min_norm_dat = min(norm_dat(:));
max_norm_dat = max(norm_dat(:));
resampledImagesOut1 = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
resampledImagesOut1 = cast(resampledImagesOut1, 'single');

if(sizeIn > 2)
    resampledImagesOut2a = double(resampledImagesIn2(startY:endY,startX:endX,:));
    
    %% JAS start
    %norm_dat = interpft(resampledImagesOut2a, 20, 3);
    norm_dat = interp3(x,y,z, (resampledImagesOut2a),x1,y1,z1);
    min_norm_dat = min(norm_dat(:));
    max_norm_dat = max(norm_dat(:));

    resampledImagesOut2 = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
    resampledImagesOut2 = cast(resampledImagesOut2, 'single');

    %% JAS end

    if(sizeIn > 3)
        resampledImagesOut3a = double(resampledImagesIn3(startY:endY,startX:endX,:));
        
        %% JAS start
        %norm_dat = interpft(resampledImagesOut3a, 20, 3);
        norm_dat = interp3(x,y,z, (resampledImagesOut3a),x1,y1,z1);
        min_norm_dat = min(norm_dat(:));
        max_norm_dat = max(norm_dat(:));

        resampledImagesOut3 = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
        resampledImagesOut3 = cast(resampledImagesOut3, 'single');
        %% JAS end
    
        if(sizeIn > 4)
            resampledImagesOut4a = double(resampledImagesIn4(startY:endY,startX:endX,:));
           
            %% JAS start
            %norm_dat = interpft(resampledImagesOut4a, 20, 3);
            norm_dat = interp3(x,y,z, (resampledImagesOut4a),x1,y1,z1);
            min_norm_dat = min(norm_dat(:));
            max_norm_dat = max(norm_dat(:));

            resampledImagesOut4 = (norm_dat - min_norm_dat)/(max_norm_dat - min_norm_dat);
            resampledImagesOut4 = cast(resampledImagesOut4, 'single');
            %% JAS end
        end
    end
end


return;
