#this function reads in the .mat MRI image data (with variable no. of frames) with RR interval.
#It then interpolates the data to 40 frames with a specified acceleration factor which effectively alters the heart rate.
#the starting phase of the heart is also randomised
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
import PlotUtils
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

#reading the data in
file='/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat'
with h5py.File(file, 'r+') as f:
    print(f.keys())
    new_dat_final = f['new_dat_final']
    one_data=f[new_dat_final[0, 0]][:]

#defining an acceleration factor and defining arrays for interpolation
one_data = np.transpose(one_data, (1, 2, 0))
one_data_dims = np.shape(one_data)
nFrames = one_data_dims[2]
matrix = one_data_dims[0]
time_crop = 40
print('nFrames',nFrames)
x = y = x1 = y1 = np.linspace(0,matrix-1,matrix)
z = np.linspace(0,nFrames-1,nFrames)
print('z',z)
accel = 3
spacing = (nFrames-1)/((nFrames/accel)-1)
print('spacing',spacing)
N = (nFrames)/spacing
print('N',N)
z1 = np.linspace(0,nFrames-1,num=int(N))
print('z1',z1)
print(one_data_dims)

#interpolation
meshx1,meshy1,meshz1 = np.meshgrid(x1,y1,z1)
grid_dat = interpn((x,y,z), one_data, np.array([meshx1,meshy1,meshz1]).T)

#transposing and normalising for plotting
print('grid_dat',np.shape(grid_dat),'max grid_dat',np.amax(grid_dat))
normed_grid_dat = grid_dat/np.amax(grid_dat)

one_data = np.transpose(one_data, (2,0,1))
normed_one_data = one_data/np.amax(one_data)

#repeating the data out to 40 frames and take random starting phase
num_repetitions = np.ceil((3*time_crop)/len(z1)) #always 120 frames before cutting
print('num_repetitions',num_repetitions)
rep_normed_grid_dat  = np.tile(normed_grid_dat,(int(num_repetitions),1,1)) # in there for random starting frame
print('rep_normed_grid_dat',np.shape(rep_normed_grid_dat))

rand_start_frame = np.random.randint(0,high = time_crop - 1) 
time_crop_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + time_crop, :, :]

print('rand_start_frame',rand_start_frame,'time_crop_rand_start',np.shape(time_crop_rand_start))
print('diff between frames',np.amax(rep_normed_grid_dat[3+len(z1),:,:]-rep_normed_grid_dat[3,:,:]))
print('check if vid starts at rand frame',np.amax(time_crop_rand_start[0,:,:]-rep_normed_grid_dat[rand_start_frame,:,:]))

PlotUtils.plotVid(normed_one_data,axis=0,vmax=1)
# PlotUtils.plotVid(normed_grid_dat,axis=0,vmax=1)
# PlotUtils.plotVid(rep_normed_grid_dat,axis=0,vmax=1)
PlotUtils.plotVid(time_crop_rand_start,axis=0,vmax=1)

