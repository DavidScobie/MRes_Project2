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
import tensorflow_addons as tfa

#Get it to run on the CPU because CPU is 16GB whereas GPU is 4GB, (and we use over 6GB here)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

patient_code = "h5_slices/"

#filepath = os.path.join("../data/firs_scan/gridded/",patient_code)
filepath = os.path.join("../../BART/data/meas_rest_stack/",patient_code)

full_filepath = os.path.join("C:/PHD/MRes_project/ML_work/BART/data/meas_rest_stack/",patient_code)
#full_filepath = os.path.join("/media/sf_ML_work/BART/data/meas_ex3_stack/",patient_code)

list = os.listdir(full_filepath) # dir is your directory path
number_files = len(list)
# number_files = 4
print(number_files)

# model = load_model('../../read_DICOMS/frames_40/dlex_augment/zcr_aug_trans_var_ex_rate/hopefully_full_model/best.h5')
model = load_model('../../read_DICOMS/frames_40/dlex_augment/Royal_Free_aug_trans_var_ex_rate/var_fram_size/70epo/best_good_inp_size.h5')
model.summary()

#Read in the first dataset
# fstart = h5py.File(os.path.join(filepath,'/d1_00001.h5'),'r')['x']
fstart = h5py.File(os.path.join(filepath, 'd1_00001.h5'),'r')['x']

fstart = tf.expand_dims(fstart, axis=3)


#get the low resolution data stacked together

for i in range(number_files - 1):
    filepath2 = os.path.join(filepath,'d1_%05d.h5') % (i+2)
    fi = h5py.File(filepath2,'r')['x']

    #rotate if necessary
    fi = tfa.image.rotate(fi,np.pi)

    fi = tf.expand_dims(fi, axis=3)
    fnext_lowres = tf.concat([fstart,fi], axis=3)
    fstart = fnext_lowres

#Permute dimensions
fnext_lowres = tf.transpose(fnext_lowres, perm=[3, 0, 1, 2])

#expand dimensions
bigger_dset_lowres = tf.expand_dims(fnext_lowres, axis=4)

#Squeeze to make image
low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset_lowres, tf.float32))

#Get all of the data to be put through the model and then stacked aferwards

for i in range(number_files - 1):
    filepath2 = os.path.join(filepath,'d1_%05d.h5') % (i+2)
    fi = h5py.File(filepath2,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.transpose(fi, perm=[3, 0, 1, 2])
    bigger_dset = tf.expand_dims(fnext, axis=4)
    # print('bigger dset size',tf.shape(bigger_dset))
    # print('model input shape',model.input_shape)
    y_pred = model.predict(bigger_dset)
    #y_pred = model.predict(bigger_dset, batch_size=8)

    if i == 0:
        y_pred1 = y_pred
    fnext = tf.concat([y_pred1,y_pred], axis=0)
    y_pred1 = fnext


# convert y pred to float32 type
test_pred = tf.squeeze(tf.image.convert_image_dtype(fnext, tf.float32))


#cropping the 192 down to 128
# test_pred = test_pred[:,:,32:160,32:160]

# turn the tensorflow arrays into numpy arrays
low_res_np = tf.make_ndarray(tf.make_tensor_proto(low_res))
test_pred_np = tf.make_ndarray(tf.make_tensor_proto(test_pred))

#Plot the reconstructed frames as an animation
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0,savepath='./model_128_same_as_pap_2')
# PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0)
PlotUtils.plotVid(np.squeeze(test_pred_np[6,:,:,:]),vmin=0,vmax=1,axis=0)

#sio.savemat('Royal_Free_aug_trans_var_ex_rate_var_fram_size_70epo_rest_prosp.mat',{'low_res_DICOM':low_res_np, 'model_recon':test_pred_np}) #you can save as many arrays as you want

plt.show()

