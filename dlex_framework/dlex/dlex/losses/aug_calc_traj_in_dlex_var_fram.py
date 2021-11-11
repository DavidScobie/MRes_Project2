#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:36:21 2021
Data augmentation function for David 
@author: oj20
"""
import tensorflow as tf
import tensorflow_addons as tfa
import math
import tensorflow_nufft as tfft
import h5py
import tensorflow_mri as tfmr
import numpy as np
import tensorflow_probability as tfp
import scipy.io as sio

def choose_params():
  #rest or aug?
  decide = np.random.randint(2) #a random integer of either 0 or 1
  if decide == 0:
    #at rest want a bit of aug to simulate respiratory motion
    ex = np.random.uniform(low=0.0, high=0.15, size=None)
  else:
    #at exercise we want augmentation at the upper end of the scale
    ex = np.random.uniform(low=0.5, high=1.0, size=None)

  #ex=0.5
  print('ex',ex)
  accel = (1.5*ex)+1
  min_motion = max_motion = (6*ex)+0 #6 pixels is max motion
  resp_freq = (1*ex)+0.25
  
  return accel, min_motion, max_motion, resp_freq


def wrapper_augment(imagex,imagey,gpu=1,time_crop=None,augment=1):
  
  if augment == 1:
    accel, min_motion, max_motion, resp_freq = choose_params()
    # accel = 2
    # min_motion = 3
    # max_motion = 3
    # resp_freq = 2
  else:
    accel = 1
  print('accel',accel)

  if gpu==1:
      with tf.device('/gpu:0'):
          imagey = interpolate_in_time(imagey, accel = accel, time_crop=time_crop)
  else:
     imagey = interpolate_in_time(imagey, accel = accel, time_crop=time_crop)
      
  if gpu==1:
      with tf.device('/gpu:0'):
          if augment==1:
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
          else:
              x, y = training_augmentation_flow(imagey)
  else:
      if augment==1:
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
      else:
              x, y = training_augmentation_flow(imagey)

  return x, y


def training_augmentation_flow_withmotion(y,time_crop=None,min_motion_ampli=0,max_motion_ampli=0,resp_freq=1):

    ##################################################### The translation part

    motion_ampli = tf.random.uniform([1],minval=min_motion_ampli,maxval=max_motion_ampli,dtype=tf.dtypes.float32,seed=None,name=None)/(time_crop/2) #random motion amplitude inbetween specified limits
    
    displacement=[]
    for idxdisp in range(time_crop): #40
        if idxdisp%40==0:               
            transform = motion_ampli
        sin_point = 20*tf.cast(transform,tf.float32)*tf.math.sin((idxdisp/(time_crop/resp_freq))*2*tf.constant(math.pi)) #sinusoidal translation
        if idxdisp==0:
            displacement.append(tf.cast([[[0]],[[0]]],tf.float32)) #defining the first value in the translation

        displacement.append(tf.cast([[sin_point],[sin_point]],tf.float32))

    displacement=tf.stack(displacement) #shape(20,2,1) values from 0 to -20.
    transform=tf.squeeze(displacement) #shape(20,2) values from 0 to -20
    del displacement
    transform = transform[:,0]

    Zeros = tf.zeros((tf.shape(transform)[0]))
    transform = tf.expand_dims(transform,1)
    Zeros = tf.expand_dims(Zeros,1)
    transform = tf.experimental.numpy.hstack((Zeros,transform))  
    del Zeros

    y=tf.expand_dims(y, axis=3)

    y=tfa.image.translate(y,tf.cast(transform[:-1,:],tf.float32),'bilinear' ) #image:y. transaltion:transform. interpolation mode: bilinear
    del transform
    x = y #(40,192,192,1)

    #####################################Gridding onto undersampled radial trajectory
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40,13,384)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)


    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  
    del radial_weights
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)
    del traj
    del dcw_kspace

    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training
 
    y = (y - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype)) / (tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype) - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype))
    x = (x - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype)) / (tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype) - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype))
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    print('shape x ',tf.shape(x),'shape y',tf.shape(y))

    return x,y


def training_augmentation_flow(y):
    #gridding onto undersampled radial trajectory
    
    x = y #(40,192,192,1)
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)
    #tf.print('x',tf.shape(x))

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    #traj_np = tf.make_ndarray(tf.make_tensor_proto(traj))
    #sio.savemat('traj_tiny_half.mat',{'traj_tiny_half':traj_np})

    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    print('traj',tf.shape(traj))
    kspace = tfft.nufft(tf.cast(x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    
    print('kspace imag max',tf.math.reduce_max(tf.math.imag(kspace)),'kspace real max',tf.math.reduce_max(tf.math.real(kspace)))

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40,13,384)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)

    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  
    # print('dcw_kspace',tf.squeeze(dcw_kspace[0][0:5]))

    del kspace
    del radial_weights
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)
    del traj
    del dcw_kspace

    y=tf.expand_dims(y,axis=-1) #making it (40,192,192,1) to work with training
    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training

    y = (y - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype)) / (tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype) - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype))
    x = (x - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype)) / (tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype) - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype))
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    return x,y


def interpolate_in_time(one_data, accel = 1, time_crop=None):
    #Taking 40 frame data and giving it an acceleration factor for heart rate

    #if accel=1 then dont need interp just tile and choose random start frame
    if accel == 1:
        print('augmentation not happening')

        print('one_data',tf.shape(one_data))
        #Finding how many we need to repeat matrix by
        one_dat_dims = tf.shape(one_data) #(15:25,192,192)
        one_dat_frames = tf.cast(one_dat_dims[0],tf.int32) #(15:25)
        num_reps_number = tf.cast(((4*time_crop)/one_dat_frames),tf.int32) #6:11
        print('num_reps_number',num_reps_number)

        #tiling out the matrix
        to_tile = [1*(num_reps_number),1,1] #[6:11,1,1]
        rep_normed_grid_dat_2  = tf.tile(one_data,to_tile) # approx (160,192,192)
        print('rep_normed_grid_dat_2',tf.shape(rep_normed_grid_dat_2))
        del one_data

        #choosing random starting frame
        rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) #0 to 39 some number
        #rand_start_frame = 0
        time_crop_rand_start = rep_normed_grid_dat_2[rand_start_frame:rand_start_frame + time_crop, :, :] # (40,192,192)
    else:
        print('augmentation is happening')
        one_data = tf.transpose(one_data, perm=[1, 2, 0]) #make it (192,192,n)
        
        one_data_dims = tf.shape(one_data) #(192,192,n)

        #defining values for interpolation
        nFrames_int32 = tf.cast(one_data_dims[2],tf.int32) #n
        matrix = tf.cast(one_data_dims[0],tf.int32) #192
        x = y = x1 = y1 = tf.linspace(0,matrix-1,matrix) #[0,1,2,...,190,191]
        z = tf.linspace(0,nFrames_int32-1,nFrames_int32) #[0,1,2,...,n-2,n-1]
        nFrames_float64 = tf.cast(nFrames_int32,tf.float64) #n
        spacing = (nFrames_float64-1)/((nFrames_float64/accel)-1) #(n-1)/((n/accel)-1)
        N = tf.cast(((nFrames_float64)/spacing),tf.int32) #n/spacing

        #tensorflow interpolation in time
        x_ref_min = tf.cast(0,tf.float64) #0
        x_ref_max = tf.cast(nFrames_int32-1,tf.float64) #n-1
        z1 = tf.linspace(0,nFrames_int32-1,num=N) #[0,...,n-1]  N points

        grid_dat = tfp.math.interp_regular_1d_grid(tf.cast(z1,tf.float64), x_ref_min, x_ref_max, tf.cast(one_data,tf.float64), axis=-1) #(192,192,6:35) ish

        del one_data

        normed_grid_dat = (grid_dat - tf.math.reduce_min(grid_dat)) / (tf.math.reduce_max(grid_dat) - tf.math.reduce_min(grid_dat)) #normalize

        grid_dat_shape = tf.shape(grid_dat) #(192,192,6:35)
        grid_dat_frames = tf.cast(grid_dat_shape[2],tf.int32) #6:35
        num_reps_number = tf.cast(((3*time_crop)/grid_dat_frames),tf.int32) #3:20
        del grid_dat

        normed_grid_dat = tf.transpose(normed_grid_dat, perm=[2,0,1]) #(6:35,192,192)
        print('normed_grid_dat',tf.shape(normed_grid_dat))

        to_tile = [num_reps_number,1,1] #[3:20,1,1]
        print('to_tile',to_tile)

        rep_normed_grid_dat  = tf.tile(normed_grid_dat,to_tile) # approx (120,192,192). Can be (108:120,192,192)
        print('rep_normed_grid_dat',tf.shape(rep_normed_grid_dat))
        del normed_grid_dat

        #giving data a random starting frame
        rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) #0:39
        time_crop_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + time_crop, :, :] #[40,192,192]
        del rep_normed_grid_dat
    print('time_crop_rand_start shape',tf.shape(time_crop_rand_start))

    return time_crop_rand_start

####################

#Quick Test
import h5py
import time
import PlotUtils
import matplotlib.pyplot as plt

#fdata = '/host/data/SAX/Royal_Free/RF_full_set_var_temp_len/'
#fdata = '/host/data/SAX/yonly/GOSH_var_temp_len/GOSH_full_set_var_temp_len/'
fdata = '/home/david/shared/read_DICOMS/data/GOSH_SAX_data/GOSH_full_set_var_temp_len/'

dataplot='val_00001'
filename=fdata + dataplot + '.h5'

ori= h5py.File(filename, 'r')
image=ori['y']

time_crop = 40

start=time.time()

x,y=wrapper_augment(image,image,gpu=0,time_crop=time_crop,augment=1)
duration=time.time()-start
print('duration',duration)

x_np = tf.make_ndarray(tf.make_tensor_proto(x))
y_np = tf.make_ndarray(tf.make_tensor_proto(y))
#sio.savemat('GOSH_full_set_var_temp_len_val1x_and_y_without_aug2_dias_start_tiny_half_dens_as_well.mat',{'y':y_np,'x':x_np})

imgx=np.concatenate((x,y),axis=1)
PlotUtils.plotVid(imgx,axis=0,vmax=1)

#check if nans in the data

# vector = tf.math.is_nan(x)
# #print(vector)
# print(np.max(vector))

# np_vector = vector.numpy()
# #print(np_vector)
# if np_vector.any():
#     print('here!')

# vector2 = tf.math.is_nan(y)
# #print(vector)
# print(np.max(vector2))

# np_vector2 = vector2.numpy()
# #print(np_vector)
# if np_vector2.any():
#     print('here2!')





