#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
# import mapvbvd
from bart import bart
from matplotlib import pyplot as plt
import cfl
import PlotUtils
import scipy.io as sio
from scipy.fft import fft, ifft
import tensorflow_mri as tfmr
# In[2]:


"""
Load Data 
"""
contents = sio.loadmat('/media/sf_ML_work/BART/meas_MID00573_FID48984_rest_stack_raw_and_ang.mat')
# `contents` is a dictionary, so access the variables like:
raw_data = contents['raw_data'] #shape (384, 13, 40, 26, 12)
raw_data_shape = np.shape(raw_data)
print('raw_data',raw_data_shape)

noSlices = int(raw_data_shape[4])
matrix = int((raw_data_shape[0])/2)
nPhases = int(raw_data_shape[2])
accSpokes = int(raw_data_shape[1])
nCoils = int(raw_data_shape[3])
# fullfile='/media/sf_ML_work/BART/rawData_exercise/meas_MID00573_FID48984_rest_stack.dat'

# twixObj = mapvbvd.mapVBVD(fullfile)
# sizeObj = len(twixObj)
# twixObj[sizeObj-1].image.flagRemoveOS = False

# data_hdr = twixObj[sizeObj-1].hdr

# noSlices = int(data_hdr.Config.NSlc)
# matrix = int(data_hdr.Config.ImageColumns)
# nPhases = int(data_hdr.Config.NPhs)
# accSpokes = int(data_hdr.Config.RadialViews)

# del data_hdr
print(noSlices)
"12"
print(matrix) 
"192"
print(nPhases) 
"40"
print(accSpokes) 
"13"

"--------------------------------------"
"  data_hdr = twixObj[sizeObj-1].hdr "
"current size 384 26 13 1 12 1 40"

# raw_data = np.squeeze(twixObj[sizeObj-1].image[:,:,:,:,:,:,:,:,:,:])
# print("raw_data size = ", raw_data.shape)
" (384, 26, 13, 12, 40) "

"----------------------------------------------"

# if noSlices == 1 :
#     raw_data = np.permute(raw_data, [0, 1, 2, 4, 3])

# nCoils = len(raw_data[1])
print(" nCoils = ", nCoils)

"""
% raw_data is currently stored in the format of:
%    matrix*2 (due to 2x OverSampling in readout directiom);
%    nCoils
%    accSpokes
%    noSlices
%    nPhases

% so in this case [384, 26, 13, 12, 40]

% reorder here just so that i can do gridding per 2D slice on each coil
% seperately
%data_image is now to be stored in the format of:
%    matrix*2 (due to 2x OverSampling in readout directiom);
%    accSpokes
%    nPhases
%    nCoils
%    noSlices

% so in this case [384, 13, 40, 26, 12]
"""
# raw_data = np.transpose(raw_data, [0, 2, 4, 1, 3])
print("raw_data = ", raw_data.shape)


# In[3]:


"now calculate the trajectory - get the radial angles from the raw data file"
"------------------------------------------------------"

# uint16Angle0 = np.uint16(twixObj[sizeObj-1].image.iceParam[:,4])  #why are there 4 angles?
# uint16Angle1 = np.uint16(twixObj[sizeObj-1].image.iceParam[:,5])
# uint16Angle2 = np.uint16(twixObj[sizeObj-1].image.iceParam[:,6]) 
# uint16Angle3 = np.uint16(twixObj[sizeObj-1].image.iceParam[:,7]) 

# del twixObj
# del sizeObj

# tt1=np.stack((uint16Angle0, uint16Angle1, uint16Angle2, uint16Angle3)) #the 4 angles get stacked together
# print("\n len(uint16Angle0) = ", len(uint16Angle0), " tt1= ", tt1.shape)

# del uint16Angle0
# del uint16Angle1
# del uint16Angle2
# del uint16Angle3

# radialAngles = []
# for i in range(accSpokes*nPhases) : #range 520
#     tt4 = np.array(tt1[:, i], dtype=np.uint16)
#     radialAngles.append(tt4.view(np.double)) #radial angles is what we make from the angles of the .dat file

# del tt1
# del tt4
# del i #the temporary ones get deleted
radialAngles = contents['radialAngles']
radialAngles = np.transpose(radialAngles, (1, 0)) #permute the dimensions
print('radial angles',np.shape(radialAngles),radialAngles.dtype)
radialAngles = radialAngles.tolist() #convert to list for the loop later
print('list radial angles',np.shape(radialAngles))
""" 
  print("ii = ", i, " , ", tt4, " , ", combinedAngle[i]) 
"""

import math
" now create trajectory as used in the sequence, to match the above data "
    
npoints = accSpokes * (matrix*2) * nPhases; # (13x192x2x40) = 199680, why do we multiply this by 13?
dimensions  = 3

trajectory = np.zeros((dimensions, int(matrix*2), int(accSpokes), int(nPhases))) #(3,384,13,40) this is number of points
" weights    = np.zeros((matrix*2, accSpokes, nPhases))"

for phs in range(nPhases) : #40
    for lin in range(accSpokes) : #13 
        ang = radialAngles[phs*accSpokes + lin] #(0:39 x 13) + (0:12), goes up to 519
        # print('ang',ang,np.shape(ang))
        # print('phs',phs,'lin',lin)
        cos_angle = math.cos(np.float64(ang))
        sin_angle = math.sin(np.float64(ang))
             
        for col in range(int(matrix)*2) : #384
            kx = (col - matrix) /2  #(0:383 - 192) /2 because of the 2x oversampling
                
            trajectory[0, col, lin, phs]=(cos_angle*kx) #fill kx up with angles. (-192:192) 
            trajectory[1, col, lin, phs]=(sin_angle*kx) #(-192:192) 
            trajectory[2, col, lin, phs]=0.0 #kz = 0
            """            
            if kx == 0.0 :
                weights[col, lin, phs]= 0.25 
            else :
                weights[col, lin, phs]=  abs(kx) 
            """
del sin_angle
del cos_angle
del col 
del phs
del lin  
del kx       


# In[4]:


GRIDresult = np.zeros((matrix, matrix, nPhases, noSlices)) # (192,192,40,12)

noSlices = 1
for sl in range(noSlices) : #range 12
    sliceData = raw_data[:,:,:,:,sl] #[384, 13, 40, 26, 1] , why do we have the 13 in here? 

    print("sliceData ", sliceData.shape)
    dataCoilSensitivities = np.zeros((matrix*2, accSpokes*nPhases, 1, nCoils), dtype=complex) #(384,520,1,26) each of the k space points has its own sensitivity!
    trajScale = np.zeros((dimensions, matrix*2, accSpokes*nPhases)) #(3,384,520)

    " calculate coil sensitivities "
    " this stacks up the data over all time points, as we do a temporal average to get a fully sampled equivalent image to calculate the CS from"
    # nPhases = 1 #REMOVE THIS LATER
    for ph in range(nPhases) : #40
        dataCoilSensitivities[:,ph*accSpokes:(ph+1)*accSpokes, 0, :] = np.squeeze(sliceData[:,:,ph,:]) #remove the [...,1] dimension from the end of sliceData. 
        # we are just reshaping here effectively
        trajScale[:,:,ph*accSpokes:(ph+1)*accSpokes] = trajectory[:,:,:, ph] #saving the trajectory of each slice as 3D matrix instead of 4D

    dataCS_3124 = np.transpose(dataCoilSensitivities, [2, 0, 1, 3]) #permute and save raw data as another variable before deletion
    del dataCoilSensitivities
    del ph

    "grid the data"
    "David: you need to chnage the following line to your gridder!!!!"
    print('trajScale',np.shape(trajScale),'dataCS_3124',np.shape(dataCS_3124))

    import tensorflow_nufft as tfft
    import tensorflow as tf
    # import tensorflow_mri as tfmr
    
    #reshape the raw data
    dataCS_perm = tf.transpose(dataCS_3124, perm=[3,0,1,2])
    print('look at this shape', tf.shape(dataCS_perm) )
    dataCS_perm =  tf.reshape(dataCS_perm , [nCoils, -1])
    print('dataCS_perm',tf.shape(dataCS_perm))

    #reshape the trajectory data
    trajSc = trajScale[0:2,:,:] #making it (2,384,520)
    print('traj_scale',tf.shape(trajSc))
    trajSc= tf.transpose(trajSc, perm=[1,2,0])
    trajSc =  tf.reshape(trajSc , [-1 , 2])
    trajSc = tf.expand_dims(trajSc,axis=0)
    trajSc = tf.repeat(trajSc, repeats = 26, axis = 0)
    # weights = tfmr.estimate_density(trajSc, (192,192)) #REMOVE THIS

    #scale from -pi to pi
    trajSc = (trajSc / 192) *2*np.pi
    print('trajSc',tf.shape(trajSc), 'max traj', np.max(trajSc), 'min traj', np.min(trajSc))

    #calculate weights
    weights = tfmr.estimate_density(trajSc, (192,192))
    print('weights shape',np.shape(weights))
    dcw_dataCS_perm = dataCS_perm / tf.cast(weights,dtype=tf.complex128)
    print('dcw_dataCS_perm', np.shape(dcw_dataCS_perm))

    print('TYPES', 'dcw_dataCS_perm', dcw_dataCS_perm.dtype, 'trajSc', trajSc.dtype)
    average_gridded_data = tfft.nufft(dcw_dataCS_perm , trajSc, transform_type='type_1', fft_direction='backward', grid_shape=(192,192))
    # average_gridded_data = tf.math.reduce_sum(average_gridded_data,axis = 1)
    print('average_gridded_data',tf.shape(average_gridded_data), 'max avg', np.max(average_gridded_data), 'min avg', np.min(average_gridded_data))
    average_gridded_data= tf.transpose(average_gridded_data, perm=[1,2,0]) #big set
    print("average_gridded_data = ", average_gridded_data.shape)
    # average_gridded_data = bart(1,"nufft -i -d"+str(matrix)+":"+str(matrix)+":1", trajScale, dataCS_3124) #The bart nufft
    " size (192x192x1x26)"

    " this part is fine "
    fig, ax = plt.subplots(nrows=1,ncols=4, figsize=(10,10))
    for cl in range(4) :
       ax[cl].imshow(abs(average_gridded_data[:,:,cl]))
    "  "
    del dataCS_3124
    average_gridded_data = tf.expand_dims(average_gridded_data,axis=2)
    print("average_gridded_data exp dims = ", average_gridded_data.shape)
    #ksp = fft(average_gridded_data.numpy())
    print('average_gridded_data dtype',average_gridded_data.dtype)
    ksp = tfmr.fftn(np.squeeze(average_gridded_data.numpy()), shape=(192,192,26), axes=None, norm='backward', shift=True)
    print('post ft ksp=', np.shape(ksp))
    fig, ax = plt.subplots(nrows=1,ncols=4,figsize=(10,10))
    for cl in range (4):
        ax[cl].imshow(abs(ksp[:,:,cl]))
    #ksp = bart(1, "fft -u 7", average_gridded_data.numpy()) #The coil sensitivities are found by taking the fourier transform of the sum over the images
    del average_gridded_data

    #coil_sensitivities = bart(1, "caldir 20", ksp) #this is a calibration of the sensitivities
    coil_sensitivities = tfmr.estimate_coil_sensitivities(ksp, method='espirit',num_maps = 1)
    " coil_sensivitoes = np.squeeze(coil_sensitivities) "
    del ksp
    " plot coil sensitivities"
    print("sl = ", sl, " , cS size = ", coil_sensitivities.shape)
    fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(10,10))
    for cl in range(5) :
        ax[cl].imshow(abs(coil_sensitivities[:,:,cl,0]))
    " "
    
#     " ---------------------------------------------- "
#     "now grid the data"

    " slice data is currently [matrix*2, accSpokes, nCoils, nPhases] "
 
    noSlices = 1 #loop over the 12 of them later
    ab_squeeze = np.zeros((40,192,192)) #for plotting at end
    nPhases = 3 #the number of temporal frames looking at
    fig, ax = plt.subplots(nrows=1, ncols=nPhases,figsize=(10,10))
    for i in range (nPhases):

        dataCoilSensitivities = np.zeros((matrix*2, accSpokes, 1, nCoils), dtype=complex) #(384,13,1,26) 
        trajScale = np.zeros((dimensions, matrix*2, accSpokes)) #(3,384,13)

        dataCoilSensitivities[:,:, 0, :] = np.squeeze(sliceData[:,:,i,:])
        trajScale[:,:,:] = trajectory[:,:,:, i] 

        trajSc = trajScale[0:2,:,:] 
        trajSc= tf.transpose(trajSc, perm=[1,2,0])
        trajSc =  tf.reshape(trajSc , [-1 , 2])
        trajSc = tf.expand_dims(trajSc,axis=0)
        trajSc = tf.repeat(trajSc, repeats = 26, axis = 0)
        trajSc = (trajSc / 192) *2*np.pi #scale between -pi and pi
        print('trajSc',tf.shape(trajSc), 'max traj', np.max(trajSc), 'min traj', np.min(trajSc)) #trajSc (26,4992,2) here
        
        #plt.figure(i+3)
        #plt.imshow()
        # print('data_coil sensitivities', np.shape(dataCoilSensitivities), 'trajScale', np.shape(trajScale))

        first_time_data = tf.transpose(dataCoilSensitivities, perm=[2,0,1,3]) #(26,13,384)
        dataCS_perm = tf.transpose(first_time_data, perm=[3,0,1,2])
        first_time_data = tf.reshape( dataCS_perm, [26 , -1]) #(26,4992) to enter to nufft yes

        weights = tfmr.estimate_density(trajSc, (192,192))
        dcw_first_time_data = first_time_data / tf.cast(weights,dtype=tf.complex128)
        print('dcw_first_time_data', np.shape(dcw_first_time_data))
        
        # print('first_time_data', tf.shape(first_time_data), 'trajSc', tf.shape(trajSc),'trajSc_type',trajSc.dtype) 
        ave_gridded_data = tfft.nufft(tf.cast(dcw_first_time_data,tf.complex128) , tf.cast(trajSc,tf.float64), transform_type='type_1', fft_direction='backward', grid_shape=(192,192))
        # print('ave_gridded_data', tf.shape(ave_gridded_data))

        ab_squeeze[i,:,:] = abs(np.squeeze(ave_gridded_data[0,:,:])) #looking at a particular coil
        ax[i].imshow(np.squeeze(ab_squeeze[i,:,:])) 

    plt.figure(200)
    plt.imshow(np.sum(ab_squeeze,axis=0))
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




