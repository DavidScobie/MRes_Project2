function [data_out] = grid_data(pos,data,weight,matrix_size,fixed_dims,over_sampling,kernel_width,kernel_beta)
%
% [data_out] = grid_data(pos,data,weight,matrix_size,fixed_dims)
%
% Basic gridding (incl. deapodization filter)
%
% INPUT:
%       -pos         : Position in k (k-t) space with respect to the grid, [npoints,ndims]
%       -data        : Complex data for points [npoints]
%       -weight      : Weights for the data points [npoints]
%       -matrix_size : Size of reconstruction matrix [ndims]
%       -fixed_dims  : Mask indicating if any of the dimensions are already
%                      on the grid, i.e. [0 0 1] means grid dimensions 1
%                      and 2 but leave 3 alone.
%
% OUTPUT:
%       -data_out    : Gridded in k (k-t) space.
%

int_over_sampling = 2.0;
int_kernel_width = 4;
int_kernel_beta = 18.5547;

%Derfault is to grid all dimensions
if nargin<5,
    fixed_dims = zeros(size(matrix_size));
end

if nargin>5,
    int_over_sampling = over_sampling;
end
    
if nargin>6,
    int_kernel_width = kernel_width;
end

if nargin>7,
    int_kernel_beta = kernel_beta;
end

%Setting some dimensions
matrix_size_oversamp = matrix_size.*int_over_sampling;
idx = find(matrix_size == 1);
matrix_size_oversamp(idx) = 1;
idx = find(fixed_dims == 1);
matrix_size_oversamp(idx) = matrix_size(idx);

[data_out] = grid_wrapper(pos,data,matrix_size,weight,1,fixed_dims,int_over_sampling,int_kernel_width,int_kernel_beta);
data_out = reshape(data_out,matrix_size_oversamp);
data_out = ktoi(data_out);
%showimage(data_out);return;

%Deapodization
if (int_over_sampling == 1 && (matrix_size(1) < 256)),
extra_oversamp = 2.0;
else
extra_oversamp = 1;
end

co = zeros(1,size(pos,2));
[imgfilter] = grid_wrapper(co, [1], matrix_size,[1],1,fixed_dims,int_over_sampling*extra_oversamp,int_kernel_width,int_kernel_beta);
imgfilter = reshape(imgfilter,matrix_size_oversamp);
imgfilter = ktoi(imgfilter) .* prod(size(imgfilter));

%Remove oversampling

if (size(pos,2) == 3),
imgfilter = imgfilter([1:size(data_out,1)]+bitshift(size(imgfilter,1)-size(data_out,1),-1), ...
                        [1:size(data_out,2)]+bitshift(size(imgfilter,2)-size(data_out,2),-1), ...
                        [1:size(data_out,3)]+bitshift(size(imgfilter,3)-size(data_out,3),-1));
else
imgfilter = imgfilter([1:size(data_out,1)]+bitshift(size(imgfilter,1)-size(data_out,1),-1), ...
                        [1:size(data_out,2)]+bitshift(size(imgfilter,2)-size(data_out,2),-1));    
end

%showimage(imgfilter,[2 2 1]);showimage(angle(imgfilter),[2 2 2]);
%showimage(itok(imgfilter),[2 2 3]); caxis(caxis .* 0.1)

%DWS
%subplot(2,2,1); imagesc(abs(imgfilter)); subplot(2,2,2); imagesc(angle(imgfilter)); 
%subplot(2,2,3); imagesc(abs(itok(imgfilter)))

data_out = data_out ./ imgfilter; 
clear imgfilter;

%DWS
%data_out=reshape(data_out,512,512);
%data_out=circshift(data_out,256);
%DWS^^

% Remove oversampling
if (size(pos,2) == 3),
    data_out = data_out([1:matrix_size(1)]-bitshift(matrix_size(1),-1)+bitshift(size(data_out,1),-1),...
                        [1:matrix_size(2)]-bitshift(matrix_size(2),-1)+bitshift(size(data_out,2),-1),...
                        [1:matrix_size(3)]-bitshift(matrix_size(3),-1)+bitshift(size(data_out,3),-1));
else    
    data_out = data_out([1:matrix_size(1)]-bitshift(matrix_size(1),-1)+bitshift(size(data_out,1),-1),...
                        [1:matrix_size(2)]-bitshift(matrix_size(2),-1)+bitshift(size(data_out,2),-1));    
    
end

%Go back to  k (k-t) space
data_out = itok(data_out);

%DWS
% data_out=circshift(data_out,128);

return