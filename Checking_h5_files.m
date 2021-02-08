clear all
close all

addpath C:\PHD\MRes_project\ML_work\dlex_framework\dlex\matlab
%%Checking x and y outputs from the model (in the cache folder)

%Loading data in
data=dlexload('C:\PHD\MRes_project\ML_work\mapped_docker_files\ml\results\final_unproc_saxalldata_small\pred\val');

%Collecting data for 1st validation image
data1=data(1);
y1=data1.y;
y1_permuted=permute(y1,[2 3 1]);

figure;
subplot(2,3,1)
imagesc(y1_permuted(:,:,1));
title('slice 1, y')

x1=data1.x;
x1_permuted=permute(x1,[2 3 1]);
subplot(2,3,2)
imagesc(x1_permuted(:,:,1));
title('slice 1, x')

y_pred_1=data1.y_pred;
y_pred_permuted=permute(y_pred_1,[2 3 1]);
subplot(2,3,3)
imagesc(y_pred_permuted(:,:,1));
title('slice 1, y pred')

%Collecting data for 4th validation image
data4=data(4);
y4=data4.y;
y4_permuted=permute(y4,[2 3 1]);

subplot(2,3,4)
imagesc(y4_permuted(:,:,1));
title('slice 4, y')

x4=data4.x;
x4_permuted=permute(x4,[2 3 1]);

subplot(2,3,5)
imagesc(x4_permuted(:,:,1));
title('slice 4, x')

y_pred_4=data4.y_pred;
y_pred_4_permuted=permute(y_pred_4,[2 3 1]);
subplot(2,3,6)
imagesc(y_pred_4_permuted(:,:,1));
title('slice 4, y pred')
%%

