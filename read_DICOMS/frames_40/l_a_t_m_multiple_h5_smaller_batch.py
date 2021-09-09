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
import os

#Get it to run on the CPU because CPU is 16GB whereas GPU is 4GB, (and we use over 6GB here)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

patient_code = "ex3_ds/"

filepath = os.path.join("../data/firs_scan/gridded/",patient_code)
# filepath = os.path.join("../data/gridded/",patient_code)

full_filepath = os.path.join("/media/sf_ML_work/read_DICOMS/data/firs_scan/gridded",patient_code) #virtual box
# full_filepath = os.path.join("C:/PHD/MRes_project/ML_work/read_DICOMS/data/firs_scan/gridded",patient_code)

list = os.listdir(full_filepath) # dir is your directory path
number_files = len(list)
# number_files = 4
print(number_files)

model = load_model('./fi_SAX_rest_and_trans_aug_40_y_only_chk_new_gridder/model2.h5')
# model = load_model('../multiple_orientations_20/fi_2dssim_optim_mse_L2_mul_ori/model.h5')
model.summary()

#Read in the first dataset
# fstart = h5py.File(os.path.join(filepath,'/d1_00001.h5'),'r')['x']
fstart = h5py.File(os.path.join(filepath, 'd1_00001.h5'),'r')['x']
fstart = tf.expand_dims(fstart, axis=3)

"""
#get the low resolution data stacked together

for i in range(number_files - 1):
    filepath2 = os.path.join(filepath,'d1_%05d.h5') % (i+2)
    fi = h5py.File(filepath2,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext

#Permute dimensions
fnext = tf.transpose(fnext, perm=[3, 0, 1, 2])

#expand dimensions
bigger_dset = tf.expand_dims(fnext, axis=4)

#Image one of the slices
low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))

"""

#Get all of the data to be put through the model and then stacked aferwards

for i in range(number_files - 1):
    filepath2 = os.path.join(filepath,'d1_%05d.h5') % (i+2)
    fi = h5py.File(filepath2,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    print('fi',tf.shape(fi),fi.dtype)

    fnext = tf.transpose(fi, perm=[3, 0, 1, 2])
    bigger_dset = tf.expand_dims(fnext, axis=4)
    y_pred = model.predict(bigger_dset)
    print('y_pred',tf.shape(y_pred),y_pred.dtype)

    if i == 0:
        y_pred1 = y_pred
    fnext = tf.concat([y_pred1,y_pred], axis=0)
    y_pred1 = fnext



    
print('fnext',tf.shape(fnext),fnext.dtype)
# fstart = fnext

#Permute dimensions
# print('fnext',tf.shape(fnext),fnext.dtype)

# print('perm fnext', tf.shape(fnext),fnext.dtype)

#expand dimensions

# print('bigger_dset',tf.shape(bigger_dset),bigger_dset.dtype)



test_pred = tf.squeeze(tf.image.convert_image_dtype(fnext, tf.float32))


#cropping the 192 down to 128
# test_pred = test_pred[:,:,32:160,32:160]

#Image one of the slices
# low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))


# turn the tensorflow arrays into numpy arrays
# low_res_np = tf.make_ndarray(tf.make_tensor_proto(low_res))
test_pred_np = tf.make_ndarray(tf.make_tensor_proto(test_pred))

#Plot the reconstructed frames as an animation
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='./model_128_same_as_pap_2')
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0)
PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0)

#save the matrices to files
# # sio.savemat('prosp1_DICOM_MSErecon_RDSSIMrecon.mat',{'low_res_DICOM':low_res_np, 'MSE_recon':test_pred_np,  'RDSSIM_recon':test_pred_np_2}) #you can save as many arrays as you want

# sio.savemat('fi_SAX_rest_and_trans_aug_40_y_only_chk_new_gridder_ex3_ds_new.mat',{'low_res_DICOM':low_res_np, 'model_recon':test_pred_np}) #you can save as many arrays as you want

plt.show()


