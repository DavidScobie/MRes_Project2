#this function reads in the .mat MRI image data (with variable no. of frames) with RR interval.
#It then interpolates the data to 40 frames with a specified acceleration factor which effectively alters the heart rate.
#the starting phase of the heart is also randomised
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import PlotUtils
import matplotlib.pyplot as plt

file='/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat'
with h5py.File(file, 'r+') as f:
    print(f.keys())
    new_dat_final = f['new_dat_final']
    one_data=f[new_dat_final[0, 0]][:]

one_data = np.transpose(one_data, (1, 2, 0))
one_data_dims = np.shape(one_data)
nFrames = one_data_dims[2]
print('nFrames',nFrames)
x = y = x1 = y1 = np.linspace(0,191,192)
z = np.linspace(0,nFrames-1,nFrames)
print('z',z)
accel = 2
spacing = (nFrames-1)/((nFrames/accel)-1)
print('spacing',spacing)
N = (nFrames)/spacing
print('N',N)
z1 = np.linspace(0,nFrames-1,num=int(N))
print('z1',z1)
print(one_data_dims)
my_interpolating_function = RegularGridInterpolator((x, y, z), one_data)
#pts = (x1,y1,z1)
pts = ([[92,96,10.7],[3,5,7]])
gridded_dat = my_interpolating_function(pts)
print(gridded_dat)
print(np.shape(gridded_dat))
print(one_data[92,96,10],one_data[92,96,11]) #the interpolator semas to be working. Just need to put all points through

#this seems like a very slow way of doing it. but check it works before optimising
grid_dat = np.zeros((192,192,8))
for i in range (192):
    for j in range (192):
        for k in range (8):
            grid_dat[i,j,k] = my_interpolating_function([x1[i],y1[j],z1[k]])

grid_dat = np.transpose(grid_dat, (2,0,1))
print('grid_dat',np.shape(grid_dat),'max grid_dat',np.amax(grid_dat))
normed_grid_dat = grid_dat/np.amax(grid_dat)

one_data = np.transpose(one_data, (2,0,1))
normed_one_data = one_data/np.amax(one_data)

PlotUtils.plotVid(normed_one_data,axis=0,vmax=1)
PlotUtils.plotVid(normed_grid_dat,axis=0,vmax=1)
#plt.show()
"""
print('x1',np.shape(x1),'y1',np.shape(y1),'z1',np.shape(z1))
end_grid = np.meshgrid(x1,y1,z1)
arr_end_grid = np.array(end_grid)
print(np.shape(arr_end_grid))
print(arr_end_grid[0:1,0:5,0:5,0:1])
"""


