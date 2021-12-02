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


gens = tf.random.Generator.from_seed(1234, alg='philox') 

def choose_params(gens):
  #rest or aug?
  # decide = np.random.randint(2) #a random integer of either 0 or 1
  # tf.print('tf decide',decide)

  #a random dimensionless tensor of either 0 or 1
  uniform_seed=gens.uniform(shape=(1, 1),minval=0, maxval=2, dtype=tf.int32)
  dim_less_uni = uniform_seed[0][0]
  tf.print('uni seed',dim_less_uni)

  @tf.function
  def zero_or_1(tensor):
    for c in tensor:        
        if tf.equal(c , 0):
            ex = gens.uniform(shape=(1, 1),minval=0, maxval=0.15, dtype=tf.float32)
            accel = (1*ex)+1
        else:
            ex = gens.uniform(shape=(1, 1),minval=0.3, maxval=1, dtype=tf.float32)
            accel = (1*ex)+1
    return ex, accel
  ex, accel = zero_or_1(uniform_seed)
  print('ex',ex,'accel',accel)

  # if decide == 0:
  #   #at rest want a bit of aug to simulate respiratory motion
  #   ex = np.random.uniform(low=0.0, high=0.15, size=None)
  #   accel = (1*ex)+1 #up to 2x rest rate
  # else:
  #   #at exercise we want augmentation at the upper end of the scale
  #   ex = np.random.uniform(low=0.3, high=1.0, size=None)
  #   accel = (1*ex)+1 #up to 2x rest rate

  #ex=0.1 #REMOVE THIS AFTER
  tf.print('tf ex',ex)

  trans_motion_ampli = (6*ex)+0 #6 pixels is max motion. spatial res = 1.67mm
  resp_freq = (1*ex)+0.25 #10-15 breaths/min up to 40-50 breaths/min. A scan is 1.5 seconds. Temp res = 36.4ms
  
  return accel, trans_motion_ampli, resp_freq


def wrapper_augment(imagex,imagey,gpu=1,time_crop=None,augment=1):
  
  if augment == 1:
    accel, trans_motion_ampli, resp_freq = choose_params(gens)
    # accel = 2
    # min_motion = 3
    # max_motion = 3
    # resp_freq = 2
  else:
    accel = 1
  print('accel',accel)
      
  if gpu==1:
      with tf.device('/gpu:0'):
          if augment==1:
            imagey = interpolate_in_time_with_RHR(imagey, accel = accel, time_crop=time_crop)
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,trans_motion_ampli=trans_motion_ampli,resp_freq=resp_freq) #CHANGE AFTER
            #x, y = training_augmentation_flow(imagey)
          else:
            imagey = interpolate_in_time_no_RHR(imagey, time_crop=time_crop)
            x, y = training_augmentation_flow(imagey)
  else:
      if augment==1:
            imagey = interpolate_in_time_with_RHR(imagey, accel = accel, time_crop=time_crop)
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,trans_motion_ampli=trans_motion_ampli,resp_freq=resp_freq) #CHANGE AFTER
            #x, y = training_augmentation_flow(imagey)
      else:
            imagey = interpolate_in_time_no_RHR(imagey, time_crop=time_crop)
            x, y = training_augmentation_flow(imagey)

  tf.debugging.check_numerics(x, message='Checking x') #throws an error if x contains a NaN or inf value
  tf.debugging.check_numerics(y, message='Checking y')

  return x, y


def training_augmentation_flow_withmotion(y,time_crop=None,trans_motion_ampli=0,resp_freq=1):

    ##################################################### The translation part

    print('trans_motion_ampli',trans_motion_ampli)

    displacement=[]
    for idxdisp in range(time_crop): #40
        if idxdisp%40==0:               
            transform = trans_motion_ampli
        sin_point = tf.cast(transform,tf.float32)*tf.math.sin((idxdisp/(time_crop/resp_freq))*2*tf.constant(math.pi)) #sinusoidal translation. max of 1cm translation at peak exercise
        if idxdisp==0:
            displacement.append(tf.cast([[[0]],[[0]]],tf.float32)) #defining the first value in the translation

        displacement.append(tf.cast([[[sin_point]],[[sin_point]]],tf.float32))

    displacement=tf.stack(displacement) #shape(20,2,1) values from 0 to -20.
    transform=tf.squeeze(displacement) #shape(20,2) values from 0 to -20
    del displacement
    transform = transform[:,0]
    
    Zeros = tf.zeros((tf.shape(transform)[0]))
    transform = tf.expand_dims(transform,1)
    Zeros = tf.expand_dims(Zeros,1)
    transform = tf.experimental.numpy.hstack((Zeros,transform))  
    del Zeros
    print('y pre trans',tf.shape(y))
    y=tf.expand_dims(y, axis=3)

    y=tfa.image.translate(y,tf.cast(transform[:-1,:],tf.float32),'bilinear' ) #image:y. transaltion:transform. interpolation mode: bilinear
    del transform

    #####################################Gridding onto undersampled radial trajectory

    x = y #(40,192,192,1)
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40,13,384)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)

    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  

    del kspace
    del radial_weights
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)
    del traj
    del dcw_kspace

    print('y after x und',tf.shape(y),'x after x und',tf.shape(x))
    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training

    y = (y - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype)) / (tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype) - tf.cast(tf.reduce_min(tf.abs(y)),dtype=y.dtype))
    x = (x - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype)) / (tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype) - tf.cast(tf.reduce_min(tf.abs(x)),dtype=x.dtype))
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    return x,y


def training_augmentation_flow(y):
    #gridding onto undersampled radial trajectory

    x = y #(40,192,192)
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 

    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny_half', angle_range='full', readout_os=2.0) #(40,13,384)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)

    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  

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


def interpolate_in_time_no_RHR(one_data, time_crop=None):
    #Taking variable frame number data. Tiling to many frames and picking a window of 40 frames at random start point.

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
    print('time_crop_rand_start shape',tf.shape(time_crop_rand_start))
    return time_crop_rand_start


def interpolate_in_time_with_RHR(one_data, accel = 1, time_crop=None):
    #Taking 40 frame data and giving it an acceleration factor for heart rate

    print('augmentation is happening')
    one_data = tf.transpose(one_data, perm=[1, 2, 0]) #make it (192,192,n)

    accel = tf.cast(accel,tf.float64)
    
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
    z1 = tf.linspace(0,nFrames_int32-1,num=N[0][0]) #[0,...,n-1]  N points

    grid_dat = tfp.math.interp_regular_1d_grid(tf.cast(z1,tf.float64), x_ref_min, x_ref_max, tf.cast(one_data,tf.float64), axis=-1) #(192,192,6:35) ish

    del one_data

    #normalise
    normed_grid_dat = (grid_dat - tf.math.reduce_min(grid_dat)) / (tf.math.reduce_max(grid_dat) - tf.math.reduce_min(grid_dat)) #normalize

    #tile the data out
    grid_dat_shape = tf.shape(grid_dat) #(192,192,6:35)
    grid_dat_frames = tf.cast(grid_dat_shape[2],tf.int32) #6:35
    num_reps_number = tf.cast(((3*time_crop)/grid_dat_frames),tf.int32) #3:20
    del grid_dat

    normed_grid_dat = tf.transpose(normed_grid_dat, perm=[2,0,1]) #(6:35,192,192)
    print('normed_grid_dat',tf.shape(normed_grid_dat))

    tf.print('size of normed_grid_dat',tf.shape(normed_grid_dat))

    to_tile = [num_reps_number,1,1] #[3:20,1,1]
    print('to_tile',to_tile)

    rep_normed_grid_dat  = tf.tile(normed_grid_dat,to_tile) # approx (120,192,192). Can be (108:120,192,192)
    print('rep_normed_grid_dat',tf.shape(rep_normed_grid_dat))
    del normed_grid_dat

    #giving data a random starting frame
    rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) #0:39
    #rand_start_frame = 0
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
print('image shape',tf.shape(image))
time_crop = 40

start=time.time()

x,y=wrapper_augment(image,image,gpu=0,time_crop=time_crop,augment=1)
duration=time.time()-start
print('duration',duration)

x_np = tf.make_ndarray(tf.make_tensor_proto(x))
y_np = tf.make_ndarray(tf.make_tensor_proto(y))
#sio.savemat('GOSH_full_set_var_temp_len_val1x_and_y_RHR_aug_only_ex0p1.mat',{'y':y_np,'x':x_np})
print('x',tf.shape(x),'y',tf.shape(y))
imgx=np.concatenate((x,y),axis=1)
PlotUtils.plotVid(imgx,axis=0,vmax=1)
