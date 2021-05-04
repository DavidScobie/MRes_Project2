# load and evaluate a saved model
from numpy import loadtxt
import tensorflow as tf
import dlex
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import PlotUtils
from PlotUtils import *
import scipy.io as sio

# load model
# model = load_model('./old_network/model.h5')
model = load_model('./model.h5')
# summarize model.
model.summary()

#Read in the first dataset of 12
fstart = h5py.File('./low_res_data/d1_00001.h5','r')['x']
fstart = tf.expand_dims(fstart, axis=3)

#Read in and concatenate the next 11 datasets
for i in range(11):
    filepath = './low_res_data/d1_%05d.h5' % (i+2)
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
# plt.figure(1)
# plt.imshow(test_pred[3,5,:,:], cmap='gray')


# turn the tensorflow arrays into numpy arrays
low_res_np = tf.make_ndarray(tf.make_tensor_proto(low_res))
test_pred_np = tf.make_ndarray(tf.make_tensor_proto(test_pred))

#Plot the reconstructed frames as an animation
PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='rdssim_network')
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='.\old_network\original')

#save the matrices to files
sio.savemat('DICOM_and_reconstruction.mat',{'low_res_DICOM':low_res_np, 'reconstruction':test_pred_np}) #you can save as many arrays as you want

plt.show()



