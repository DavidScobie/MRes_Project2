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
import os

#Read in the first dataset
patient_code = "pat_6/"
file_start = "./scanner_reconstruction/"



fstart = h5py.File(os.path.join(file_start, patient_code, 'd1_00001.h5'),'r')['y_pred']
# fstart = h5py.File('./scanner_reconstruction/pat_3/d1_00001.h5','r')['y_pred']
fstart = tf.expand_dims(fstart, axis=3)

full_filepath = os.path.join(file_start,patient_code)
list = os.listdir(full_filepath) # dir is your directory path
number_files = len(list)
print(number_files)


#Read in and concatenate the next n datasets
for i in range(number_files - 2):
    filepath = os.path.join(file_start, patient_code, 'd1_%05d.h5') % (i+2)
    # filepath = filepath % (i+2)
    # filepath = './scanner_reconstruction/pat_3/d1_%05d.h5' % (i+2)
    fi = h5py.File(filepath,'r')['y_pred']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext

#Permute dimensions
fnext = tf.transpose(fnext, perm=[3, 0, 1, 2])

#expand dimensions
bigger_dset = tf.expand_dims(fnext, axis=4)

#Image one of the slices
# low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))
# plt.figure(0)
# plt.imshow(low_res[3,5,:,:], cmap='gray')

scan_recon = tf.squeeze(tf.image.convert_image_dtype(fnext, tf.float32))

scan_recon = scan_recon/(tf.math.reduce_max(scan_recon))
scan_recon_np = tf.make_ndarray(tf.make_tensor_proto(tf.transpose(scan_recon, perm=[0, 3, 1, 2])))
# scan_recon_np = tf.make_ndarray(tf.make_tensor_proto(scan_recon))

PlotUtils.plotVid(np.squeeze(scan_recon_np[6,:,:,:]),vmin=0,vmax=1,axis=0)

sio.savemat('scanner_recon_pat_6.mat',{'scanner_recon':scan_recon_np}) #you can save as many arrays as you want

plt.show()
