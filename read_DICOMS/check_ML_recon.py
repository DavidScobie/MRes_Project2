from numpy import loadtxt
import tensorflow as tf
import dlex
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import PlotUtils
from PlotUtils import *
import scipy.io as sio

#Read in the first dataset of 11
fstart = h5py.File('./ML_recon/d1_00001.h5','r')['y_pred']
fstart = tf.expand_dims(fstart, axis=3)

#Read in and concatenate the next 11 datasets
for i in range(10):
    filepath = './ML_recon/d1_%05d.h5' % (i+2)
    fi = h5py.File(filepath,'r')['y_pred']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext

#Permute dimensions
fnext = tf.transpose(fnext, perm=[3, 0, 1, 2])
#expand dimensions
bigger_dset = tf.expand_dims(fnext, axis=4)

#Image one of the slices
ML_rec = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))

ML_rec_np = tf.make_ndarray(tf.make_tensor_proto(ML_rec))

PlotUtils.plotVid(np.squeeze(ML_rec_np[6,:,:,:]),vmin=0,vmax=1,axis=0)
# PlotUtils.plotVid(np.squeeze(ML_rec_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='.\ML_recon_vid')