import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmr
import tensorflow_nufft as tfft
import time
import PlotUtils
import scipy.io as sio

#fdata = '/host/data/SAX/Royal_Free/RF_full_set_2/'
fdata = '/host/data/frames_40_all_a_few/'

dataplot='val_00001'
filename=fdata + dataplot + '.h5'

with h5py.File(filename, 'r') as f:
    y = f['y'][...]
print(y.shape)

# mask = tf.ones([128, 128])
# mask = tf.pad(mask, [[32, 32], [32, 32]])
# y *= mask
# y = tf.cast(y, tf.complex64)

# Calculate trajectory and weights.
kwargs = dict(
    base_resolution=192, views=13, phases=40, ordering='sorted', angle_range='full', readout_os=2.0
)

traj = tfmr.radial_trajectory(**kwargs) #(40, 13, 384, 2) 
traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2], traj.shape[3]]) #(40, 4992,2)

# traj = tf.reshape(traj, [-1, 384, 2])
# for i, line in enumerate(traj[:6]):
#     plt.plot(line[:, 0], line[:, 1], label=str(i))
#     plt.legend()
# plt.show()

radial_weights = tfmr.radial_density(**kwargs) #(40,13,384)
radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)

# Sample k-space.
kspace = tfft.nufft(y, traj, transform_type='type_2', fft_direction='forward') #(40,4992)

# Load real k-space.



dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(40,4992)  
x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)


y=tf.math.abs(y)
x=tf.math.abs(x)

y=y/tf.reduce_max(y)
x=x/tf.reduce_max(x)

x_np = tf.make_ndarray(tf.make_tensor_proto(x))
y_np = tf.make_ndarray(tf.make_tensor_proto(y))

imgx=np.concatenate((x,y),axis=1)
sio.savemat('val1x_and_y.mat',{'y':y_np,'x':x_np})

PlotUtils.plotVid(imgx,axis=0,vmax=1)
