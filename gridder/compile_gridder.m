current = pwd;

%cd c:\msh\mrprogs\matlab\gridder
%cd c:\Documents and ~Settings\Jenny\Desktop\conj_

%DWS
cd C:\PHD\MRes_project\Basic_tests\MATLAB\gridder\gridder
% cd c:\MATLAB\gridder\gridder

mex grid_wrapper.c grid.c
% mex grid.c

cd(current);