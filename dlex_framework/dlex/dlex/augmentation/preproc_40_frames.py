#this function reads in the .mat MRI image data (with variable no. of frames) with RR interval.
#It then interpolates the data to 40 frames with a specified acceleration factor which effectively alters the heart rate.
#the starting phase of the heart is also randomised
import h5py
import numpy as np
# SAXALLDATA = h5py.File('/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat', 'r')
# print(SAXALLDATA.keys())
# print(type(SAXALLDATA["new_dat_final"][:]))
# print(SAXALLDATA["new_dat_final"][...])
# SAXALLDATA["new_dat_final"][:]
# new_dat_final = np.array(SAXALLDATA["new_dat_final"][:])
# print(new_dat_final)

# import hdf5storage
# mat = hdf5storage.loadmat('/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat')
# print(np.shape(mat["new_dat_final"][0]))

import h5py
file='/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat'
with h5py.File(file, 'r+') as f:
    print(f.keys())
    new_dat_final = f['new_dat_final']
    ab=f[new_dat_final[0, 0]][:]
print(np.shape(ab))
print(np.amax(ab))



# import numpy as np
# import h5py
# # f = h5py.File('somefile.mat','r')
# data = SAXALLDATA.get('new_dat_final')
# print(data)
# data = np.array(data) # For converting to a NumPy array
# print(data)

# hf = h5py.File('path/to/file.h5', 'r')
# n1 = np.array(hf["dataset_name"][:]) #dataset_name is same as hdf5 object name 

# rr_int = SAXALLDATA['rr_int']

# n_d_f_0 = new_dat_final[0]
# print(n_d_f_0)
# rr_0 = rr_int[0]
# print(rr_0)



