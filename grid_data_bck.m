function [data_out] = grid_data_bck(pos,data,fixed_dims,over_sampling,kernel_width,kernel_beta)

%  [data_out] = grid_data_bck(pos,data,weight,matrix_size,fixed_dims)
%
%  Resampling on arbitrary trajectory
%
% INPUT:
%      -pos        : Positions in k (k-t) space to be sampled. Relative to the
%                    grid. [npoints,ndims].
%      -data       : Cartesian k-space [kx,ky,....]
%      -fixed_dims : Mask indicating dimensions that need no interpolation,
%                    i.e. [0 0 1] means interpolate along dimensions 1 and
%                    2 but not along 3.


%Setting some parameters
weight = [];
matrix_size = size(data);

%It is standard to interpolate along all dimensions
if nargin<3,
    fixed_dims = zeros(size(matrix_size));
end

int_over_sampling = 2.0;
int_kernel_width = 4;
int_kernel_beta = 18.5547;

if nargin>3,
    int_over_sampling = over_sampling;
end
    
if nargin>4,
    int_kernel_width = kernel_width;
end

if nargin>5,
    int_kernel_beta = kernel_beta;
end

%Gridding parameters
OverSampling = int_over_sampling;


%Setting some dimensions
matrix_size_oversamp = matrix_size.*OverSampling;
idx = find(matrix_size == 1);
matrix_size_oversamp(idx) = 1;
idx = find(fixed_dims == 1);
matrix_size_oversamp(idx) = matrix_size(idx);


img = zeros(matrix_size_oversamp);
img_ft = ktoi(data);

if (size(pos,2) == 3),
    img([1:matrix_size(1)]-bitshift(matrix_size(1),-1)+bitshift(size(img,1),-1),...
        [1:matrix_size(2)]-bitshift(matrix_size(2),-1)+bitshift(size(img,2),-1),...
        [1:matrix_size(3)]-bitshift(matrix_size(3),-1)+bitshift(size(img,3),-1)) = img_ft;clear img_ft;
else
    img([1:matrix_size(1)]-bitshift(matrix_size(1),-1)+bitshift(size(img,1),-1),...
        [1:matrix_size(2)]-bitshift(matrix_size(2),-1)+bitshift(size(img,2),-1)) = img_ft;clear img_ft;
end

%Deapodization filter
co = zeros(1,size(pos,2));
[imgfilter] = grid_wrapper(co, [1], matrix_size,[1],1,fixed_dims,int_over_sampling,int_kernel_width,int_kernel_beta);
imgfilter = reshape(imgfilter,size(img));
imgfilter = ktoi(imgfilter) .* prod(size(imgfilter));

%DWS
% imgfilter=reshape(imgfilter,512,512);
% imgfilter=circshift(imgfilter,256);
%imgfilter=circshift(imgfilter,sqrt(length(imgfilter)).*0.5);
%figure; imagesc(abs(img)); title('img');
%figure; imagesc(abs(imgfilter)); title('imgfilter_new');
%DWS^^

img = img ./ imgfilter; clear imgfilter;
img = itok(img);     

[data_out] = grid_wrapper(pos, img, matrix_size,[],0,fixed_dims,int_over_sampling,int_kernel_width,int_kernel_beta);

return