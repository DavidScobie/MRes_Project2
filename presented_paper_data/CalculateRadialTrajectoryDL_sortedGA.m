function [trajectory, weights] = CalculateRadialTrajectoryDL_sortedGA(col_len, phs_len, acc_fact)

%       col_len is matrix size (assumes it is square)
%       nRadialSpokesFS is no of fully sampled radial spokes 
%       phs_len is no of phases (time_points)
%       bIsGA is a bool used to determine if it is golden angle data 

if(col_len ~= 192)
    disp('check calculation for nRadials');
end

    nRadialSpokesFS     = calculateNoRadials(192, 192, 320);                  % assumes no of spiral required to foill k-space is the same as the matrix size
    nRadialSpokes_ACC   = floor(nRadialSpokesFS / acc_fact)
	
    disp(['nRadialSpokes_ACC should be 13, = ', int2str(nRadialSpokes_ACC)])
    nRadialSpokesFS = nRadialSpokes_ACC*acc_fact;
    
    npoints     = nRadialSpokes_ACC * col_len * phs_len; 
    dimensions  = 3;
 
    trajectory  = zeros(npoints, dimensions);  % kx, ky, t
	weights     = zeros(npoints, 1);

%Jens code
tau = (1 + sqrt(5)) / 2;
ga = pi / tau;
 
N = nRadialSpokes_ACC; % number of spokes per frame
M = phs_len; % number of frames
    
	for (phs = 1 : phs_len)
        
        anglesJ = linspace((phs-1)*nRadialSpokes_ACC, phs*nRadialSpokes_ACC-1, nRadialSpokes_ACC) * ga;
        anglesJ = anglesJ - 2 * pi * floor(anglesJ ./ (2*pi));
        anglesJ = sort(anglesJ);
        
        for (lin = 1 : nRadialSpokes_ACC)
			ang = anglesJ(lin);
            
            cos_angle = cos(ang);
            sin_angle = sin(ang);
             
            for (col = 1: col_len)  
                kx = (col-1) - (col_len/2);

                p = (phs-1)*nRadialSpokes_ACC*col_len + (lin-1)*col_len + col; % over_sampling
                
				trajectory(p, 1) = cos_angle*kx;
				trajectory(p, 2) = sin_angle*kx;						
                trajectory(p, 3) = (phs-1) - floor(phs_len/2);
                %disp('CHECK TRAJ PHS VALUE')
                        
				if(kx == 0.0)
					weights(p, 1) = 0.25;
                else
					weights(p, 1) = abs(kx);
                end
            end
        end    
    end

   cols = 20
   p = colormap(hsv(cols)) 
 
   for (phs = 1)% : phs_len)
       figure;
       pos = mod(phs,cols);
       if(pos == 0) pos = 1; end
            
       curp=p(pos, :);
       for(i=1:nRadialSpokes_ACC)
           hold on;plot(trajectory((phs-1)*nRadialSpokes_ACC*col_len + (col_len*(i-1))+1: (phs-1)*nRadialSpokes_ACC*col_len + (col_len*i),1), trajectory((phs-1)*nRadialSpokes_ACC*col_len + (col_len*(i-1))+1: (phs-1)*nRadialSpokes_ACC*col_len + (col_len*i), 2), 'color', p(pos, :));
       end
       pause(0.1);
           
%       F(phs) = getframe();
   end
    
   title('sortGA')
%    figure;
%   movie(F)
        
return