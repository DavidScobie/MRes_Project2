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
import time
import tensorflow_nufft as tfft
from scipy.interpolate import interpn
import pdb
import h5py
import tensorflow_mri as tfmr
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def choose_params():
  #exercise parameter used to make accel, freq and amplitude
  ex = np.random.uniform(low=0.0, high=1.0, size=None)
  #ex = 0
  accel = (1.5*ex)+1
  min_motion = max_motion = (6*ex)+0 #6 pixels is max motion
  resp_freq = (1*ex)+0.25
  
  # return accel
  return accel, min_motion, max_motion, resp_freq


def wrapper_augment(imagex,imagey,
                    gpu=1,time_crop=None,
                    central_crop=192, grid_size=[192,192],augment=1
                    ):
  
  if augment == 1:
    accel, min_motion, max_motion, resp_freq = choose_params()
    # accel = 2
    # min_motion = 3
    # max_motion = 3
  else:
    accel = 1
  print('accel',accel)

  imagey = interpolate_in_time(imagey, accel = accel, time_crop=time_crop)
      
  if gpu==1:
      with tf.device('/gpu:0'):
          if augment==1:
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
          else:
              x, y = training_augmentation_flow(imagey,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size)
  else:
      if augment==1:
            x, y = training_augmentation_flow_withmotion(imagey,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
      else:
              x, y = training_augmentation_flow(imagey,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size)

  return x, y


#image_label is (image_y, image_x) which are both y in this case
def training_augmentation_flow_withmotion(image_label,time_crop=None,central_crop=128,grid_size=[192,192],min_motion_ampli=0,max_motion_ampli=0,resp_freq=1):

    ##################################################### The translation part
    y= image_label
    #print('y shape',tf.shape(y))

    #random motion amplitude inbetween specified limits
    motion_ampli = tf.random.uniform([1],minval=min_motion_ampli,maxval=max_motion_ampli,dtype=tf.dtypes.float32,seed=None,name=None)/(time_crop/2)
    
    displacement=[]
    for idxdisp in range(time_crop): #40
        #print('idxdisp',idxdisp)
        if idxdisp%40==0:               

            transform = motion_ampli
            #print('transform at first',transform)
        sin_point = 20*tf.cast(transform,tf.float32)*tf.math.sin((idxdisp/(time_crop/resp_freq))*2*tf.constant(math.pi))
        #print('sin_point',tf.shape(sin_point),sin_point)
        #this gives you a random number between the min and max amplitudes
        if idxdisp==0:
            displacement.append(tf.cast([[[0]],[[0]]],tf.float32)) #HERE IT IS shape (2,1)

        displacement.append(tf.cast([[sin_point],[sin_point]],tf.float32)) #IT FAILS HERE

    displacement=tf.stack(displacement) #shape(20,2,1) values from 0 to -20.
    transform=tf.squeeze(displacement) #shape(20,2) values from 0 to -20
    #y size (20,192,192,2). this translates the y by the transform
    transform = transform[:,0]
    #print('transform',transform,'sum first 20',tf.reduce_sum(transform[0:20]),tf.reduce_sum(transform[21:40]))

    Zeros = tf.zeros((tf.shape(transform)[0]))
    #print('zeros',Zeros)
    transform = tf.expand_dims(transform,1)
    Zeros = tf.expand_dims(Zeros,1)
    transform = tf.experimental.numpy.hstack((Zeros,transform))  

    y=tf.expand_dims(y, axis=3)

    y=tfa.image.translate(y,tf.cast(transform[:-1,:],tf.float32),'bilinear' ) #image:y. transaltion:transform. interpolation mode: bilinear
    x = y #(40,192,192,1)
    #print('y shape',tf.shape(y),'transform shape','x shape',tf.shape(x),'transform shape')

    #####################################Gridding onto undersampled radial trajectory
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)
    xpre = x

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    #traj = tf.transpose(traj,perm=[0,2,1,3]) #(40,384,13,2)
    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(xpre,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40,13,384)
    #radial_weights = tf.transpose(radial_weights,perm=[0,2,1]) #(40,384,13)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)


    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  
    #dcw_kspace = tf.transpose(dcw_kspace,perm=[1,0]) #NO EFFECT
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)

    # xpre = tf.abs(xpre)/(tf.math.reduce_max(abs(xpre)))
    # x_im = tf.abs(x)/(tf.math.reduce_max(abs(x)))
    # imgx=np.concatenate((x_im,xpre),axis=1)

    #PlotUtils.plotVid(imgx,axis=0,vmax=1)
 

    #y=tf.expand_dims(y,axis=-1)
    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training
    
    #print('y before norm',y[0,0,0:10],'x before norm',x[0,0,0:10])

    y=y/tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype)
    x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    print('shape x ',tf.shape(x),'shape y',tf.shape(y))

    return x,y

#image_label is (image_y, image_x) which are both y in this case
def training_augmentation_flow(image_label,time_crop=None,central_crop=128,grid_size=[192,192]):

    y= image_label

    x = y #(40,192,192,1)
    print('y shape',tf.shape(y),'transform shape','x shape',tf.shape(x),'transform shape')
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)
    xpre = x

    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    #traj = tf.transpose(traj,perm=[0,2,1,3]) #(40,384,13,2)
    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(xpre,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    

    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40,13,384)
    #radial_weights = tf.transpose(radial_weights,perm=[0,2,1]) #(40,384,13)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)


    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  
    #dcw_kspace = tf.transpose(dcw_kspace,perm=[1,0]) #NO EFFECT
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)

    # xpre = tf.abs(xpre)/(tf.math.reduce_max(abs(xpre)))
    # x_im = tf.abs(x)/(tf.math.reduce_max(abs(x)))
    # imgx=np.concatenate((x_im,xpre),axis=1)

    #PlotUtils.plotVid(imgx,axis=0,vmax=1)
    """
    Crop
    """

    y=tf.expand_dims(y,axis=-1) #making it (40,192,192,1) to work with training
    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training
    
    #print('y before norm',y[0,0,0:10],'x before norm',x[0,0,0:10])

    y=y/tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype)
    x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    print('shape x ',tf.shape(x),'shape y',tf.shape(y))

    return x,y


def interpolate_in_time(one_data, accel = 1, time_crop=None):
    #Taking 40 frame data and giving it an acceleration factor for heart rate

    #if accel=1 then dont need interp just choose random start frame
    if accel == 1:
        print('augmentation not happening')
        rep_normed_grid_dat  = tf.tile(one_data,[3,1,1])
        rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) 
        time_crop_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + time_crop, :, :]
    else:
        print('augmentation is happening')
        one_data = tf.transpose(one_data, perm=[1, 2, 0]) #make it (192,192,40)
        
        one_data_dims = tf.shape(one_data)

        nFrames_int32 = tf.cast(one_data_dims[2],tf.int32)#issue is that this is zero
        matrix = tf.cast(one_data_dims[0],tf.int32)
        x = y = x1 = y1 = tf.linspace(0,matrix-1,matrix)
        z = tf.linspace(0,nFrames_int32-1,nFrames_int32)
        nFrames_float64 = tf.cast(nFrames_int32,tf.float64)
        spacing = (nFrames_float64-1)/((nFrames_float64/accel)-1)
        N = tf.cast(((nFrames_float64)/spacing),tf.int32)
        z1 = tf.linspace(0,nFrames_int32-1,num=N)

        #tensorflow interpolation in time
        x_ref_min = tf.cast(0,tf.float64)
        x_ref_max = tf.cast(nFrames_int32-1,tf.float64)
        z1 = tf.linspace(0,nFrames_int32-1,num=N)

        grid_dat = tfp.math.interp_regular_1d_grid(tf.cast(z1,tf.float64), x_ref_min, x_ref_max, tf.cast(one_data,tf.float64), axis=-1)

        normed_grid_dat = grid_dat/tf.math.reduce_max(grid_dat) #this could be where the issue is YES

        num_reps_number = ((3*time_crop)//tf.shape(z1)[0])

        normed_grid_dat = tf.transpose(normed_grid_dat, perm=[2,0,1])

        to_tile = [num_reps_number,1,1]

        rep_normed_grid_dat  = tf.tile(normed_grid_dat,to_tile) # in there for random starting frame


        #giving data a random starting frame
        rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) 
        time_crop_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + time_crop, :, :]
    print('time_crop_rand_start shape',tf.shape(time_crop_rand_start))

    return time_crop_rand_start

####################
"""
#Quick Test
import h5py
import numpy as np
import sys
# sys.path.insert(0, '/home/oj20/UCLjob/PythonCode/Utils/')
sys.path.insert(0, '/sf_ML_work/read_DICOMS/')

import PlotUtils
import time
#fdata = '/host/data/SAX/Royal_Free/SAX_Royal_Free_40_192_noise_yonly/'
fdata = '/host/data/SAX/Royal_Free/RF_full_set_2/'
#fdata = '/media/sf_ML_work/mapped_docker_files/ml/data/yonly/a_few_frames_40_rest_yonly/'
#fdata = '/media/sf_ML_work/mapped_docker_files/ml/data/yonly/Royal_Free_SAX_data/a_couple_SAX_Royal_Free_40_192_noise_yonly/'

dataplot='train_00070'
filename=fdata + dataplot + '.h5'

ori= h5py.File(filename, 'r')
image=ori['y']

#interpolate data to 40 frames with specified accleration factor and start off at random initial temporal frame
time_crop = 40
#image = interpolate_in_time(image, accel = 3, time_crop=time_crop)

naugment=1 #it seems that this determines the range that augment_counter goes up to. 
stopmean=0
#for i in range(naugment): 
start=time.time()
#mask2 and image are the same in this case (both are y) 

x,y=wrapper_augment(image,image,gpu=0,time_crop=time_crop,augment=1)
stop=time.time()-start
stopmean=stopmean+stop

    
stopmean=stopmean/naugment
#print('Mean time:',stopmean,'\n Last time:',stop)
# PlotUtils.plotXd(ys,vmax=1,vmin=0)
#print('max y',tf.math.reduce_max(y),'min y',tf.math.reduce_min(y),'type',y.dtype,'shape',tf.shape(y))
imgx=np.concatenate((x,y),axis=1)
#print('imgx',np.shape(imgx),'max imgx',np.amax(imgx))
PlotUtils.plotVid(imgx,axis=0,vmax=1)
"""

