function [image_out] = SimulatingUndersampledRadialData_sortedGA(image_in, acc_fact)

    newMatrix = 192;
    nFrames = size(image_in, 3)
    [x,y,z] = meshgrid(1:newMatrix,1:newMatrix,1:nFrames);
    [x1,y1,z1] = meshgrid(1:newMatrix,1:newMatrix, 1:(nFrames-1)/19:nFrames);

    if(size(x1, 3) ~= 20)
        disp('ERROR');
    end
    image_in = [x1,y1,z1];
%close all;
% addpath 'gridder'

    col_len = size(image_in, 1);
    phs_len = size(image_in, 3);
        
    %----------------------------------------------------------------------
   %  Calculate predicted traj + weights
    %----------------------------------------------------------------------
                                     
    [trajectory, weights] = CalculateRadialTrajectoryDL_sortedGA(col_len, phs_len, acc_fact);
    
%    disp('trajectory calculated');
    
    
        %----------------------------------------------------------------------
        % Now sample the k-space data with this trajectory
        %----------------------------------------------------------------------

        k_image_in     = itok(image_in, [1 2]);
        sampled_data   = grid_data_bck(trajectory, k_image_in, [0 0 1]);   
    
%        disp('sampled data calculated');
        %----------------------------------------------------------------------
        % Now grid all data
        %----------------------------------------------------------------------   
      % this is the bit that fails if phs in trsjectory is set to -phs/2 to
      % phs/2
      
        gridded_k_data   = grid_data(trajectory, sampled_data, weights, [col_len, col_len, phs_len], [0 0 1]);                                  
%        disp('data regridded');
    
        image_out = ktoi(gridded_k_data, [1, 2]);

        % there is some scaling problem i am not sure about.....
        image_out = image_out / (col_len/(acc_fact*10.0));
    
%        figure;imagesc(abs(image_in(:,:,1)))
%        figure;imagesc(abs(image_out(:,:,1)))


%    end
return