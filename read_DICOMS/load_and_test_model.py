# load and evaluate a saved model
from numpy import loadtxt
import tensorflow as tf
from dlex import dlex
# from tvmae_dlex_version import dlex
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import PlotUtils
from PlotUtils import *
import scipy.io as sio

# load model
# model = load_model('./old_network/model.h5')

model = load_model('./JAS_preproc_data/fi_2dssim_optim_mse_L2/model.h5')
# summarize model.
model.summary()

#Read in the first dataset
fstart = h5py.File('./data/gridded/grid_pat_4/d1_00001.h5','r')['x']
fstart = tf.expand_dims(fstart, axis=3)

#Read in and concatenate the next 11 datasets
for i in range(13):
    filepath = './data/gridded/grid_pat_4/d1_%05d.h5' % (i+2)
    fi = h5py.File(filepath,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext

#Permute dimensions
fnext = tf.transpose(fnext, perm=[3, 0, 1, 2])
#expand dimensions
bigger_dset = tf.expand_dims(fnext, axis=4)

#Image one of the slices
low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))
# plt.figure(0)
# plt.imshow(low_res[3,5,:,:], cmap='gray')

y_pred = model.predict(bigger_dset)

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))


#cropping the 192 down to 128
# test_pred = test_pred[:,:,32:160,32:160]

# turn the tensorflow arrays into numpy arrays
low_res_np = tf.make_ndarray(tf.make_tensor_proto(low_res))
test_pred_np = tf.make_ndarray(tf.make_tensor_proto(test_pred))

#Plot the reconstructed frames as an animation
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='./model_128_same_as_pap_2')
PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0)

#save the matrices to files
# # sio.savemat('prosp1_DICOM_MSErecon_RDSSIMrecon.mat',{'low_res_DICOM':low_res_np, 'MSE_recon':test_pred_np,  'RDSSIM_recon':test_pred_np_2}) #you can save as many arrays as you want
sio.savemat('fi_2dssim_optim_mse_L2_grid_pat_4.mat',{'low_res_DICOM':low_res_np, 'model_recon':test_pred_np}) #you can save as many arrays as you want

plt.show()



