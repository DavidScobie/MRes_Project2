% clear all
% close all
load('SAXdataAll.mat');
% load('smallData.mat');

%display the first slice through each of the 5 images
% for i = 1:5
%     data(i)=new_dat_final(i);
%     data_double=cell2mat(data(i));
%     u = data_double;
%     u_init=u(:,:,1);
%     figure
%     imagesc(u_init);
% end

%making animation of the first slice
data_1=new_dat_final(1);
data_double=cell2mat(data_1);
d_d_size=size(data_double);
figure
for k=1:5
    for i = 1:d_d_size(3)
        imagesc(data_double(:,:,i));
        pause(0.1)
    end
end
    
% data_1=new_dat_final(1);
% u=cell2mat(data_1);
% u_init=u(:,:,1);
% imagesc(u_init)

%Splitting up training and test data sets
% training_set=new_dat_final(1:2000);
% new_dat_final=new_dat_final(2001:2268);
new_dat_final=new_dat_final(1:2000);
%
