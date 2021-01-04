%load 'smallData.mat'
% addpath 'gridder'
acc_fact    = 13;

nVolumes = size(new_dat_final, 2)
% nVolumes = size(data_1, 2)

 for(b=1:nVolumes)

        images = new_dat_final{1, b};
%         images = data_1{1, b};
%         disp(['TEST; b = ', int2str(b), ' , nFrames = ', int2str(size(images, 3))]);
 
%         disp('tiny golden angle rotating trajectory');
%        tGA = abs(SimulatingUndersampledRadialData(images, acc_fact, true, true, true));       
%        divGS = abs(SimulatingUndersampledRadialData_dividedGS(images, acc_fact));
        
        sortGA = abs(SimulatingUndersampledRadialData_sortedGA(images, acc_fact));
             
%        [images_truth(:,:,:,b), images_tGA(:,:,:,b), images_divGS(:,:,:,b), images_sortGA(:,:,:,b)] = resample_undersample_data(images, tGA, divGS, sortGA);
        [images_truth(:,:,:,b), images_sortGA(:,:,:,b)] = resample_undersample_data(images, sortGA);

 end
 
tt = sum(sum(sum(sum(isnan(images_truth), 4), 3), 2), 1);
if(tt<0)
    disp('testData_truth ERROR contains NAN');
    return;
end
% tt = sum(sum(sum(sum(isnan(images_tGA), 4), 3), 2), 1)
% if(tt<0)
%     disp('testData_tGA_rot ERROR contains NAN');
%     return;
% end
% tt = sum(sum(sum(sum(isnan(images_divGS), 4), 3), 2), 1)
% if(tt<0)
%     disp('testData_tGA_rot ERROR contains NAN');
%     return;
% end
tt = sum(sum(sum(sum(isnan(images_sortGA), 4), 3), 2), 1);
if(tt<0)
    disp('testData_tGA_rot ERROR contains NAN');
    return;
end

save('Small_set_truth',  'images_truth');
%save('TrainData_tGA',    'images_tGA');
save('Small_set_sortGA', 'images_sortGA');
%save('TrainData_divGS',  'images_divGS');

return;
