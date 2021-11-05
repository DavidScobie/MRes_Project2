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
import scipy.io

#This script takes in a .mat file of x and y after augmentation
#I then do model.predict on x, and look at x,y and ypred

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#fdata = 'C:\PHD\MRes_project\ML_work\read_DICOMS\data\Royal_Free_SAX_data\RF_full_set_var_temp_len_y_and_x_after_aug'

patient_code = 'RF_full_set_var_temp_len_val1x_and_y_ex1'

full_filepath = os.path.join("C:/PHD/MRes_project/ML_work/read_DICOMS/data/Royal_Free_SAX_data/RF_full_set_var_temp_len_val_a_few/aug/",patient_code)

fdata = full_filepath + '.mat'

data = scipy.io.loadmat(fdata)
y = data['y']
print(np.shape(y))

x = data['x']

#model = load_model('../../read_DICOMS/frames_40/dlex_augment/Royal_Free_aug_trans_var_ex_rate/var_fram_size/without_augment/59epo/best_var_fram_size.h5')
model = load_model('../../read_DICOMS/frames_40/SAX_rest_and_trans_aug_40_y_only_tfft_gridder_tfmr061/model_64.h5')
model.summary()

x_expanded = tf.expand_dims(x, axis=0)
print('x_expanded',tf.shape(x_expanded))

y_pred = model.predict(x_expanded)
print('y_pred',tf.shape(y_pred))

x_np = tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(x_expanded)))
y_np = tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(y)))
ypred_np = tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(y_pred)))

img = np.concatenate((x_np,y_np,ypred_np),axis=2)

PlotUtils.plotVid(img,vmin=0,vmax=1,axis=0)
# PlotUtils.plotVid(x_np,vmin=0,vmax=1,axis=0)
# PlotUtils.plotVid(y_np,vmin=0,vmax=1,axis=0)
# PlotUtils.plotVid(ypred_np,vmin=0,vmax=1,axis=0)

#sio.savemat('Royal_Free_aug_trans_var_ex_rate_var_fram_size_without_augment_59epo_RF_full_set_var_temp_len_without_aug_val1x_y_ypred.mat',{'x':x_np, 'y':y_np, 'y_pred':ypred_np}) #you can save as many arrays as you want
sio.savemat('SAX_rest_and_trans_aug_40_y_only_tfft_gridder_tfmr061_RF_full_set_var_temp_len_val1x_and_y_ex1_ypred.mat',{'x':x_np, 'y':y_np, 'y_pred':ypred_np}) #you can save as many arrays as you want