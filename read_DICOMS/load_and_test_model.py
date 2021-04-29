# load and evaluate a saved model
from numpy import loadtxt
import tensorflow as tf
import dlex
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import PlotUtils
from PlotUtils import *

# load model
model = load_model('model.h5')
# summarize model.
model.summary()

# filename = './low_res_data/d1_00001.h5'
# k = tf.keras.utils.HDF5Matrix(filename)
fstart = h5py.File('./low_res_data/d1_00001.h5','r')['x']
print(fstart)
fstart = tf.expand_dims(fstart, axis=3)
# f2 = h5py.File('./low_res_data/d1_00002.h5','r')['x']
# print(f2)
# stck = tf.stack([fstart,f2], axis=3)
# print(stck)
# empty = tf.zeros([20, 192, 192, 12], tf.float32)

for i in range(11):
    print(i)
    filepath = './low_res_data/d1_%05d.h5' % (i+2)
    fi = h5py.File(filepath,'r')['x']
    fi = tf.expand_dims(fi, axis=3)
    fnext = tf.concat([fstart,fi], axis=3)
    fstart = fnext
    # f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
print(fnext)

'''
bigger_dset = tf.expand_dims(dset, axis=0)
bigger_dset = tf.expand_dims(bigger_dset, axis=4)

low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))
plt.figure(0)
plt.imshow(low_res[0,:,:], cmap='gray')

y_pred = model.predict(bigger_dset)
print(y_pred.shape)

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
print(low_res)
plt.figure(1)
plt.imshow(test_pred[0,:,:], cmap='gray')


# PlotUtils.plotVid(img,vmin=0,vmax=1,axis=0)

plt.show()
'''


