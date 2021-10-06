#this function reads in the .mat MRI image data (with variable no. of frames) with RR interval.
#It then interpolates the data to 40 frames with a specified acceleration factor which effectively alters the heart rate.
#the starting phase of the heart is also randomised
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import PlotUtils
import matplotlib.pyplot as plt

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
print('nFrames',nFrames)
x = y = x1 = y1 = np.linspace(0,191,192)
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
my_interpolating_function = RegularGridInterpolator((x, y, z), one_data)

#interpolating over time
grid_dat = np.zeros((192,192,len(z1)))
for i in range (192):
    for j in range (192):
        for k in range (len(z1)):
            grid_dat[i,j,k] = my_interpolating_function([x1[i],y1[j],z1[k]])

#transposing and normalising for plotting
grid_dat = np.transpose(grid_dat, (2,0,1))
print('grid_dat',np.shape(grid_dat),'max grid_dat',np.amax(grid_dat))
normed_grid_dat = grid_dat/np.amax(grid_dat)

one_data = np.transpose(one_data, (2,0,1))
normed_one_data = one_data/np.amax(one_data)

#repeating the data out to 40 frames and take random starting phase
num_repetitions = np.ceil(120/len(z1)) #always 120 frames before cutting
print('num_repetitions',num_repetitions)
rep_normed_grid_dat  = np.tile(normed_grid_dat,(int(num_repetitions),1,1)) # in there for random starting frame
print('rep_normed_grid_dat',np.shape(rep_normed_grid_dat))

rand_start_frame = np.random.randint(0,high=39) 
frame_40_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + 40, :, :]
print('rand_start_frame',rand_start_frame,'frame_40_rand_start',np.shape(frame_40_rand_start))

print('diff between frames',np.amax(rep_normed_grid_dat[7,:,:]-rep_normed_grid_dat[2,:,:]))

print('check if vid starts at rand frame',np.amax(frame_40_rand_start[0,:,:]-rep_normed_grid_dat[rand_start_frame,:,:]))

PlotUtils.plotVid(normed_one_data,axis=0,vmax=1)
PlotUtils.plotVid(normed_grid_dat,axis=0,vmax=1)
PlotUtils.plotVid(rep_normed_grid_dat,axis=0,vmax=1)
PlotUtils.plotVid(frame_40_rand_start,axis=0,vmax=1)

