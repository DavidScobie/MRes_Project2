#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
# import mapvbvd
from bart import bart
from matplotlib import pyplot as plt
import cfl
import PlotUtils
import scipy.io as sio
import tensorflow_mri as tfmr
import tensorflow_nufft as tfft
import tensorflow as tf
import math

contents = sio.loadmat('/media/sf_ML_work/BART/meas_MID00573_FID48984_rest_stack_raw_and_ang.mat') #CHANGE
raw_data = contents['raw_data'] #shape (384, 13, 40, 26, 12)
raw_data_shape = np.shape(raw_data)

noSlices = int(raw_data_shape[4])
matrix = int((raw_data_shape[0])/2)
nPhases = int(raw_data_shape[2])
accSpokes = int(raw_data_shape[1])
nCoils = int(raw_data_shape[3])

radialAngles = contents['radialAngles']
del contents
radialAngles = np.transpose(radialAngles, (1, 0)) #permute the dimensions
radialAngles = radialAngles.tolist() #convert to list for the loop later

dimensions  = 3

trajectory = np.zeros((dimensions, int(matrix*2), int(accSpokes), int(nPhases))) #(3,384,13,40) this is number of points

for phs in range(nPhases) : #40
    for lin in range(accSpokes) : #13 
        ang = radialAngles[phs*accSpokes + lin] #(0:39 x 13) + (0:12), goes up to 519
        cos_angle = math.cos(np.float64(ang))
        sin_angle = math.sin(np.float64(ang))
             
        for col in range(int(matrix)*2) : #384
            kx = (col - matrix) /2  #(0:383 - 192) /2 because of the 2x oversampling                
            trajectory[0, col, lin, phs]=(cos_angle*kx) #fill kx up with angles. (-192:192) 
            trajectory[1, col, lin, phs]=(sin_angle*kx) #(-192:192) 
            trajectory[2, col, lin, phs]=0.0 #kz = 0

del sin_angle
del cos_angle
del col 
del phs
del lin  
del kx  
del radialAngles 
del ang   

noSlices = 1 #CHANGE
starting_slice = 7 #CHANGE
coil_corr = np.zeros((noSlices,nCoils,nPhases,matrix,matrix),dtype = 'complex_')

for sl in range(noSlices) : #range 12
    sl = sl+starting_slice 
    print('sl',sl)

    """Finding the coil sensitivities"""

    sliceData = raw_data[:,:,:,:,sl] #[384, 13, 40, 26, 12] , why do we have the 13 in here? 

    dataCoilSensitivities = np.zeros((matrix*2, accSpokes*nPhases, 1, nCoils), dtype=complex) #(384,520,1,26) each of the k space points has its own sensitivity!
    trajScale = np.zeros((dimensions, matrix*2, accSpokes*nPhases)) #(3,384,520)

    for ph in range(nPhases) : #40
        dataCoilSensitivities[:,ph*accSpokes:(ph+1)*accSpokes, 0, :] = np.squeeze(sliceData[:,:,ph,:]) #remove the [...,1] dimension from the end of sliceData. 
        # we are just reshaping here effectively
        trajScale[:,:,ph*accSpokes:(ph+1)*accSpokes] = trajectory[:,:,:, ph] #saving the trajectory of each slice as 3D matrix instead of 4D

    dataCS_3124 = np.transpose(dataCoilSensitivities, [2, 0, 1, 3]) #permute and save raw data as another variable before deletion
    del dataCoilSensitivities
    del ph

    dataCS_perm = tf.transpose(dataCS_3124, perm=[3,0,1,2])
    del dataCS_3124
    dataCS_perm =  tf.reshape(dataCS_perm , [nCoils, -1])

    #reshape the trajectory data
    trajSc = trajScale[0:2,:,:] #making it (2,384,520)
    del trajScale
    trajSc= tf.transpose(trajSc, perm=[1,2,0])
    #trajSc= tf.transpose(trajSc, perm=[2,1,0]) #this is wrong!
    trajSc =  tf.reshape(trajSc , [-1 , 2])
    trajSc = tf.expand_dims(trajSc,axis=0)
    trajSc = tf.repeat(trajSc, repeats = 26, axis = 0)

    #scale from -pi to pi
    trajSc = (trajSc / 192) *2*np.pi

    #calculate weights
    weights = tfmr.estimate_density(trajSc, (192,192))
    print('weights shape',np.shape(weights), 'max wei', np.max(weights), 'min wei', np.min(weights))

    radial_weights = tfmr.radial_density(13*192, views=1, phases=40, spacing='sorted', domain='full', readout_os=2.0)
    print('radial_weights',np.shape(radial_weights), 'max rad_wei', np.max(radial_weights), 'min rad_wei', np.min(radial_weights)) #(40,1,4992)
    radial_weights = tf.transpose(radial_weights, perm=[1,0,2]) #(1,40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)
    radial_weights = tf.repeat(radial_weights, repeats = 26, axis = 0) #(26,199680)

    dcw_dataCS_perm = dataCS_perm / tf.cast(radial_weights,dtype=tf.complex128)
    print('dataCS_perm',np.shape(dataCS_perm))
    del dataCS_perm
    del weights
    average_gridded_data = tfft.nufft(dcw_dataCS_perm , trajSc, transform_type='type_1', fft_direction='backward', grid_shape=(192,192))
    del dcw_dataCS_perm
    average_gridded_data= tf.transpose(average_gridded_data, perm=[1,2,0]) #big set
    
    ksp = tfmr.fftn(np.squeeze(average_gridded_data.numpy()), shape=(192,192,26), axes=None, norm='backward', shift=True)
    del average_gridded_data

    coil_sensitivities = tfmr.estimate_coil_sensitivities(ksp, method='espirit',num_maps = 1)
    del ksp

    print('coil_sensitivities',tf.shape(coil_sensitivities))
    fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(10,10))
    for cl in range(4) :
       ax[cl].imshow(abs(coil_sensitivities[:,:,cl]))

    """Gridding the individual frames"""

    nPhases = 40 #the number of temporal frames looking at
    a_frame_img = img_squeeze = np.zeros((40,192,192),dtype = 'complex_')
    for i in range (nPhases):
        print(i)

        dataCoilSensitivities = np.zeros((matrix*2, accSpokes, 1, nCoils), dtype=complex) #(384,13,1,26) 
        trajScale = np.zeros((dimensions, matrix*2, accSpokes)) #(3,384,13)

        dataCoilSensitivities[:,:, 0, :] = np.squeeze(sliceData[:,:,i,:])
        #del sliceData
        trajScale[:,:,:] = trajectory[:,:,:, i] 

        trajSc = trajScale[0:2,:,:] #(2,384,13)
        del trajScale
        trajSc= tf.transpose(trajSc, perm=[1,2,0])
        trajSc =  tf.reshape(trajSc , [-1 , 2])
        trajSc = tf.expand_dims(trajSc,axis=0)
        trajSc = tf.repeat(trajSc, repeats = 26, axis = 0)
  
        # trajSc2 = trajSc[:,i*(matrix*2*accSpokes):(i+1)*(matrix*2*accSpokes),:] #in batches of 4992. we already calculated trajSc above for all 40 frames.
        #so now just take each of the 40 frames at a time
        trajSc = (trajSc / 192) *2*np.pi #scale between -pi and pi
        #print('trajSc2',tf.shape(trajSc), 'max traj', np.max(trajSc), 'min traj', np.min(trajSc)) #trajSc (26,4992,2) here

        first_time_data = tf.transpose(dataCoilSensitivities, perm=[2,0,1,3]) #(26,13,384)
        del dataCoilSensitivities
        first_time_data = tf.transpose(first_time_data, perm=[3,0,1,2])
        first_time_data = tf.reshape(first_time_data, [26 , -1]) #(26,4992) to enter to nufft yes

        weights = tfmr.estimate_density(trajSc, (192,192))

        radial_weights = tfmr.radial_density(13*192, views=1, phases=1, spacing='sorted', domain='full', readout_os=2.0) #(1,1,4992)
        #print('radial_weights',np.shape(radial_weights), 'max rad_wei', np.max(radial_weights), 'min rad_wei', np.min(radial_weights))
        radial_weights = tf.transpose(radial_weights, perm=[1,2,0]) #(1,4992,1)
        radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,4992)
        radial_weights = tf.repeat(radial_weights, repeats = 26, axis = 0) #(26,4992)
        
        dcw_first_time_data = first_time_data / tf.cast(radial_weights,dtype=tf.complex128)
        del weights
        del first_time_data

        ave_gridded_data = tfft.nufft(tf.cast(dcw_first_time_data,tf.complex128) , tf.cast(trajSc,tf.float64), transform_type='type_1', fft_direction='backward', grid_shape=(192,192))
        a_frame_img[i,:,:] = np.squeeze(ave_gridded_data[0,:,:]) #first coil
        # plt.figure(107)
        # plt.imshow(abs(np.squeeze(a_frame_img[i,:,:])))
        del trajSc
        del dcw_first_time_data
        for j in range (nCoils):
            coil_corr[sl-starting_slice,j,i,:,:] = np.multiply(np.squeeze(ave_gridded_data[j,:,:]), np.conjugate(np.squeeze(coil_sensitivities[:,:,j,0])))
            #coil_corr[sl-starting_slice,j,i,:,:] = np.squeeze(ave_gridded_data[j,:,:]) #without coil sensitivity
        del ave_gridded_data

plt.figure(200)
plt.imshow(abs(coil_corr[0,0,0,:,:])) #show slice 0, coil 0, 0th frame
print('coil_corr',coil_corr.dtype)

# plt.figure(201)
# plt.imshow(abs(coil_corr[1,3,37,:,:])) #show slice 1, coil 3, 37th frame
# print('coil_corr',coil_corr.dtype)

# plt.figure(202)
# plt.imshow(abs(coil_corr[0,3,37,:,:])) #show slice 0, coil 3, 37th frame
# print('coil_corr',coil_corr.dtype)

# plt.figure(203)
# plt.imshow(abs(coil_corr[0,2,37,:,:])) #show slice 0, coil 2, 37th frame

# plt.figure(204)
# plt.imshow(abs(coil_corr[0,2,38,:,:])) #show slice 0, coil 2, 37th frame

# plt.figure(205)
# plt.imshow(abs(coil_corr[1,2,38,:,:])) 

#sum over the coils
sum_coil = np.sum(coil_corr, axis=1)

plt.figure(206)
plt.imshow(abs(np.squeeze(sum_coil[0,0,:,:]))) 

# plt.figure(207)
# plt.imshow(abs(np.squeeze(sum_coil[0,14,:,:]))) 

sio.savemat('meas_MID00573_FID48984_rest_stack_sl_7.mat',{'img_data':sum_coil}) #CHANGE
plt.show()

#         """   
#         print("gridded_data2 = ", gridded_data2.shape)
#         print("coil_sensitivities = ", coil_sensitivities.shape)
#         weightedData = abs(np.multiply(gridded_data2, np.conjugate(np.squeeze(coil_sensitivities))))
#         print("weigghtedData = ", weightedData.shape)
     
#         tempData = np.sum(weightedData, axis=2) 
#         print("tempData = ", tempData.shape)
      
#         GRIDresult[:,:,ph,sl] = abs(tempData)
#         """  
#     del ph  
#     """
#     print("sl = ", sl  )
#     fig, ax = plt.subplots(nrows=5, figsize=(6,10))
#     for ph in range(5) :
#         ax[ph].imshow(np.squeeze(abs(GRIDresult[:,:,ph,sl])), vmin=0, vmax =0.001)
#     " ---------------------------------------------- "
#     """
#     del coil_sensitivities
    
# del dimensions
# del radialAngles


# In[ ]:




