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
f = h5py.File('./low_res_data/d1_00001.h5','r')
keys = f.keys()
print(keys)
dset = f['x']
print(dset)


bigger_dset = tf.expand_dims(dset, axis=0)
bigger_dset = tf.expand_dims(bigger_dset, axis=4)

low_res = tf.squeeze(tf.image.convert_image_dtype(bigger_dset, tf.float32))
plt.figure(0)
plt.imshow(low_res[7,:,:], cmap='gray')

y_pred = model.predict(bigger_dset)
print(y_pred.shape)

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
print(low_res)
plt.figure(1)
plt.imshow(test_pred[7,:,:], cmap='gray')


# PlotUtils.plotVid(img,vmin=0,vmax=1,axis=0)

# kev = open("low_res_data.csv","w+")

# # save numpy array as csv file
# from numpy import asarray
# from numpy import savetxt
# # define data
# data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# # save to csv file
# savetxt('low_res_data.csv', low_res, delimiter=',')

# def iter_3D(low_res):
#     for i in range(low_res.shape[0]):
#         for j in range(low_res.shape[1]):
#             for k in range(low_res.shape[2]):
#                 yield i, j, k

# l = []

# for i, j, k in iter_3D(low_res):
#     l.append('%d %d %d %d' %(str(indices_x(i, j, k)), str(indices_y(i, j, k)), str(indices_z(i, j, k)), str(low_res[i, j, k])))

# with open('file_1.csv', 'w') as f:
#     f.write("\n".join(l))

# import numpy as np
# import pandas as pd
# from pandas import Panel
# # create an example array
# a = np.arange(24).reshape([2,3,4])
# # convert it to stacked format using Pandas
# stacked = pd.Panel(low_res.swapaxes(1,2)).to_frame().stack().reset_index()
# stacked.columns = ['x', 'y', 'z', 'value']
# # save to disk
# stacked.to_csv('stacked.csv', index=False)

plt.show()


