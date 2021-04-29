# load and evaluate a saved model
from numpy import loadtxt
import tensorflow as tf
import dlex
import h5py
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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
plt.imshow(low_res[0,:,:])

y_pred = model.predict(bigger_dset)
print(y_pred.shape)

test_pred = tf.squeeze(tf.image.convert_image_dtype(y_pred, tf.float32))
plt.figure(1)
plt.imshow(test_pred[0,:,:])

plt.show()

# test_indices = range(191,192)

# def my_test_generator(subject_indices):
#     for iSbj in subject_indices:
#         # idx_frame_indics = range(num_subjects)
#         relevant_keys = [s for s in keys if 'frame_%04d_' % (iSbj) in s]
#         # idx_frame_indics = range(len(relevant_keys))
#         idx_frame_indics= range(4,5)
#         for idx_frame in idx_frame_indics:
#             f_dataset = 'frame_%04d_%03d' % (iSbj, idx_frame)
#             frame = tf.math.divide(tf.keras.utils.HDF5Matrix(filename, f_dataset), 255)
#             yield(tf.expand_dims(frame, axis=2))

# test_dataset = tf.data.Dataset.from_generator(generator = lambda: my_test_generator(subject_indices=test_indices), 
#                                          output_types = (tf.float32),
#                                          output_shapes = (frame_size))

# test_batch = test_dataset.shuffle(buffer_size=1024).batch(1)

# set1 = 
# model.predict('d1_00002.h5')

