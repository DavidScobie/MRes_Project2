% imshow(low_res_DICOM(1,2,:,:))
% low_res_1 = squeeze(low_res_DICOM(7,:,:,:));
scanner_recon_1 = squeeze(scanner_recon(7,:,:,:));
% RDSSIM_recon_1 = squeeze(RDSSIM_recon(6,:,:,:));

% implay(permute(scanner_recon_1,[2 3 1]))
implay(scanner_recon_1/(max(max(max(scanner_recon_1)))))