% clear all
% close all
% load('SAXdataAll.mat');
data_1=new_dat_final(1);
u=cell2mat(data_1);
u_init=u(:,:,1);
imagesc(u_init)

training_set=new_dat_final(1:2000);
test_set=new_dat_final(2001:2268);
