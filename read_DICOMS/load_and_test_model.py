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
model = load_model('model.h5')
# summarize model.
model.summary()

# filename = './low_res_data/d1_00001.h5'
# k = tf.keras.utils.HDF5Matrix(filename)
fstart = h5py.File('./low_res_data/d1_00001.h5','r')['x']
print(fstart)
fstart = tf.expand_dims(fstart, axis=3)

for i in range(11):
    print(i)
    filepath = './low_res_data/d1_%05d.h5' % (i+2)
    fi = h5py.File(filepath,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext
# print(fnext)
fnext = tf.transpose(fnext, perm=[3, 0, 1, 2])
# print(fnext)

# bigger_dset = tf.expand_dims(dset, axis=0)
bigger_dset = tf.expand_dims(fnext, axis=4)

low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))
print(low_res)
plt.figure(0)
plt.imshow(low_res[3,5,:,:], cmap='gray')

y_pred = model.predict(bigger_dset)
print(y_pred.shape)

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
plt.figure(1)
plt.imshow(test_pred[3,5,:,:], cmap='gray')


# PlotUtils.plotVid(img,vmin=0,vmax=1,axis=0)
low_res_np = tf.make_ndarray(tf.make_tensor_proto(low_res))
print(low_res_np)

PlotUtils.plotVid(np.squeeze(low_res_np[1,:,:,:]),vmin=0,vmax=1,axis=0)

#save the matrices to files
sio.savemat('DICOM_and_reconstruction.mat',{'low_res_DICOM':tf.make_ndarray(tf.make_tensor_proto(low_res)), 'reconstruction':tf.make_ndarray(tf.make_tensor_proto(test_pred))}) #you can save as many arrays as you want

plt.show()



