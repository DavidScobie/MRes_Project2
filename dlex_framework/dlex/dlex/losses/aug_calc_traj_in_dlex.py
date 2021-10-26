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

rng = tf.random.Generator.from_seed(123, alg='philox')
rng2 = tf.random.Generator.from_seed(1234, alg='philox')  
precomputedseeds=rng2.make_seeds(1000)[0]
global augment_counter
augment_counter=0
deterministic_counter=0 #reset every det_counter iterations (max 500) -> det_counter set at validation set size to have the same transform at each validation step.

def loadtrajectory(trajfile):
    # global traj
    # global dcw
    # global augment_counter
    with h5py.File(trajfile, 'r') as f:
            traj = f['traj'][...]/tf.reduce_max(f['traj'][...])*tf.constant(math.pi)
            dcw = f['dcw'][...]
    #augment_counter=1       
    traj=tf.reshape(traj,(traj.shape[0],traj.shape[1]*traj.shape[2],traj.shape[3]))
    dcw=tf.cast(tf.reshape(dcw,(dcw.shape[0],dcw.shape[1]*dcw.shape[2])),dtype=tf.complex64)
    return traj,dcw
#loadtrajectory('/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5')

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
                    gpu=1,maxrot=45.0,time_axis=2,time_crop=None,
                    central_crop=192, grid_size=[192,192],regsnr=8,deterministic=0,det_counter=10
                    ):
  
  #print('tf.math.reduce_max(imagey)',tf.math.reduce_max(imagey))
  #print('tf.math.reduce_max(imagey)[0]',tf.math.reduce_max(imagey)[0])

  #check if there is Nans in the data. If there is then make the data zeros
 
 
#   if tf.math.is_nan (tf.math.reduce_max(imagey)):
#       print('Nan in the data!!')
#       imagex = imagey = tf.zeros([40, 192, 192], tf.float32)
#       print('imagey in the nan if loop',imagey[0,0,0:10])
#   return imagex, imagey

  #imagex, imagey = function(imagey)

  accel, min_motion, max_motion, resp_freq = choose_params()
  #print('accel',accel,'resp_freq',resp_freq)
  #exercise parameter used to make accel, freq and amplitude
  #print('min mot',min_motion,'max_motion',max_motion)
#   min_motion = max_motion = 3
#   resp_freq = 2
#   accel = 2


  seed = rng.make_seeds(2)[0]
  if deterministic==1:
      global deterministic_counter
      seed=precomputedseeds[deterministic_counter*2:deterministic_counter*2+2]
      deterministic_counter=(deterministic_counter+1)%det_counter
  
  imagey = interpolate_in_time(imagey, accel = accel, time_crop=time_crop)
  #print('imagey after interpolate in time',imagey[0,0,0:10])
      
  if gpu==1:
      with tf.device('/gpu:0'):
          #if max_motion>0:
            x, y = training_augmentation_flow_withmotion(imagey, seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
          #else:
              #x, y = training_augmentation_flow(imagey, seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr)
  else:
      #if max_motion>0:
            x, y = training_augmentation_flow_withmotion(imagey, seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,min_motion_ampli=min_motion,max_motion_ampli=max_motion,resp_freq=resp_freq)
      #else:
              #x, y = training_augmentation_flow(imagey, seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr)

  return x, y


#image_label is (image_y, image_x) which are both y in this case
def training_augmentation_flow_withmotion(image_label,seed,maxrot=45.0,time_axis=2,time_crop=None,central_crop=128,grid_size=[192,192],regsnr=8,min_motion_ampli=0,max_motion_ampli=0,resp_freq=1):

    y= image_label
    #print('y shape',tf.shape(y))

    global augment_counter
    #random motion amplitude inbetween specified limits
    motion_ampli = tf.random.uniform([1],minval=min_motion_ampli,maxval=max_motion_ampli,dtype=tf.dtypes.float32,seed=None,name=None)/(time_crop/2)
    #print('motion_ampli',motion_ampli)
    #print('augment_counter',augment_counter)
    # if augment_counter%2==1: #if it is even
    
    displacement=[]
    for idxdisp in range(time_crop): #40
        #print('idxdisp',idxdisp)
        if idxdisp%40==0:               
            #transform=tf.random.stateless_uniform((1,1), seed+idxdisp, minval=-motion_ampli, maxval=motion_ampli) #shift in 1 dimension
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
    
    x=tf.squeeze(x) #make it acceptable for tfft.nufft   (40,192,192)
    xpre = x

############################ 1 frame
    # test = tf.constant([[[-11, 2], [3, 4]],[[1, 2], [3, 4]]])
    # print('max test',tf.math.reduce_max(test),'min test',-1*(tf.math.reduce_max(-test)))

    # view = 13

    # f1x_pre = x[0,:,:]
    # f1x = x[0,:,:] #(1,192,192)
    # traj = tfmr.radial_trajectory(192, views=view, phases=1, ordering='tiny_half', angle_range='full', readout_os=2.0) #(1,13,384,2)

    # print('max traj',tf.math.reduce_max(traj),'min traj',-1*(tf.math.reduce_max(-traj)))

    # traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(1, 4992,2)
    # kspace = tfft.nufft(tf.cast(f1x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(1,4992)
    # print('max kspace',tf.math.reduce_max(abs(kspace)),'min kspace',-1*(tf.math.reduce_max(abs(-kspace))))

    # radial_weights = tfmr.radial_density(192, views=view, phases=1, ordering='tiny_half', angle_range='full', readout_os=2.0) #(1,13,384)
    # print('max radial_weights',tf.math.reduce_max(abs(radial_weights)),'min radial_weights',-1*(tf.math.reduce_max(abs(-radial_weights))))

    # radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(1,4992)
    # dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,4992)
    # x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(1,192,192)
    # print('max x',tf.math.reduce_max(abs(x)),'min x',-1*(tf.math.reduce_max(abs(-x))))

    # x_im = tf.abs(x)/(tf.math.reduce_max(abs(x)))
    # xpre_im = tf.expand_dims(tf.abs(f1x_pre)/(tf.math.reduce_max(abs(f1x_pre))),axis=0)
    # imgx=np.concatenate((x_im,xpre_im),axis=1)
    # PlotUtils.plotVid(imgx,axis=0,vmax=1)

############################# CURRENT METHOD
    traj = tfmr.radial_trajectory(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40, 13, 384, 2) 
    #traj = tf.transpose(traj,perm=[0,2,1,3]) #(40,384,13,2)
    traj = tf.reshape(traj, [traj.shape[0], traj.shape[1]*traj.shape[2] , traj.shape[3]]) #(40, 4992,2)
    kspace = tfft.nufft(tf.cast(xpre,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)    
#############################  OTHER METHOD
    # traj = tfmr.radial_trajectory(192, views=13*40, phases=1, ordering='tiny', angle_range='full', readout_os=2.0) #(1,520,384,2)
    # traj = tf.reshape(traj, [40,4992,2])
    # kspace = tfft.nufft(tf.cast(x,tf.complex64), traj,transform_type='type_2', fft_direction='forward') #(40,4992)
    # kspace = tf.reshape(kspace, [1 , -1]) #(1,199680)   

############################ CURRENT METHOD NO FLASH BUT STILL BLURRED
    radial_weights = tfmr.radial_density(192, views=13, phases=40, ordering='tiny', angle_range='full', readout_os=2.0) #(40,13,384)
    #radial_weights = tf.transpose(radial_weights,perm=[0,2,1]) #(40,384,13)
    radial_weights = tf.reshape(radial_weights,[traj.shape[0], traj.shape[1]]) #(40,4992)
    radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)
########################### OTHER METHOD THIS IS WHAT GIVES THE FLASH
    # radial_weights = tfmr.radial_density(192, views=13*40, phases=1, ordering='tiny', angle_range='full', readout_os=2.0)
    # radial_weights = tf.transpose(radial_weights, perm=[1,0,2]) #(1,40,4992) CRUCIAL, IF OTHER WAY ROUND THE ORDER IS WRONG
    # radial_weights = tf.reshape(radial_weights, [1 , -1]) #(1,199680)
##########################

    dcw_kspace = kspace / tf.cast(radial_weights,dtype=tf.complex64) #(1,199680)  
    #dcw_kspace = tf.transpose(dcw_kspace,perm=[1,0]) #NO EFFECT
    dcw_kspace = tf.reshape(dcw_kspace, [40 , 4992])  

    x = tfft.nufft(dcw_kspace, traj, grid_shape=(192,192), transform_type='type_1', fft_direction='backward') #(40,192,192)
    #print('x after undersamp',np.shape(x),'max x',tf.math.reduce_max(abs(x)),'min x',tf.math.reduce_min(abs(x)))

    # xpre = tf.abs(xpre)/(tf.math.reduce_max(abs(xpre)))
    # x_im = tf.abs(x)/(tf.math.reduce_max(abs(x)))
    # imgx=np.concatenate((x_im,xpre),axis=1)

    #PlotUtils.plotVid(imgx,axis=0,vmax=1)
    """
    Crop
    """

    #y=tf.expand_dims(y,axis=-1)
    x=tf.expand_dims(x,axis=-1) #making it (40,192,192,1) to work with training
    
    #print('y before norm',y[0,0,0:10],'x before norm',x[0,0,0:10])

    y=y/tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype)
    x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)

    # if tf.math.reduce_max(y)>0: #if image is full of zeros dont want to divide by zero
    #     y=y/tf.cast(tf.reduce_max(tf.abs(y)),dtype=y.dtype)
    # else:
    #     y=y

    # if tf.math.reduce_max(abs(x))>0: #if image is full of zeros dont want to divide by zero
    #     x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)
    # else:
    #     x=x

    #print('y after norm',y[0,0,0:10],'x after norm',x[0,0,0:10])

    #y = tf.squeeze(y)

    #print('y size',tf.shape(y),'x',tf.shape(x))
    
    y=tf.math.abs(y)
    x=tf.math.abs(x)

    print('shape x ',tf.shape(x),'shape y',tf.shape(y))

    return x,y

def load_data(file=None):
    with h5py.File(file, 'r+') as f:
        #print(f.keys())
        new_dat_final = f['new_dat_final']
        one_data=f[new_dat_final[0, 0]][:]
    return one_data

def interpolate_in_time(one_data, accel = 1, time_crop=None):

    one_data = tf.transpose(one_data, perm=[1, 2, 0])
    
    one_data_dims = tf.shape(one_data)

    nFrames_int32 = tf.cast(one_data_dims[2],tf.int32)#issue is that this is zero
    matrix = tf.cast(one_data_dims[0],tf.int32)
    x = y = x1 = y1 = tf.linspace(0,matrix-1,matrix)
    z = tf.linspace(0,nFrames_int32-1,nFrames_int32)
    nFrames_float64 = tf.cast(nFrames_int32,tf.float64)
    spacing = (nFrames_float64-1)/((nFrames_float64/accel)-1)
    N = tf.cast(((nFrames_float64)/spacing),tf.int32)
    z1 = tf.linspace(0,nFrames_int32-1,num=N)

    #tensorflow interpolation
    x_ref_min = tf.cast(0,tf.float64)
    x_ref_max = tf.cast(nFrames_int32-1,tf.float64)
    z1 = tf.linspace(0,nFrames_int32-1,num=N)

    grid_dat = tfp.math.interp_regular_1d_grid(tf.cast(z1,tf.float64), x_ref_min, x_ref_max, tf.cast(one_data,tf.float64), axis=-1)
    #print('grid_dat_before_norm',grid_dat[0,0,0:10])

    normed_grid_dat = grid_dat/tf.math.reduce_max(grid_dat) #this could be where the issue is YES

    # if tf.math.reduce_max(grid_dat)>0: #if image is full of zeros dont want to divide by zero
    #     normed_grid_dat = grid_dat/tf.math.reduce_max(grid_dat) #this could be where the issue is YES
    # else: 
    #     normed_grid_dat = grid_dat

    #print('grid_dat_after_norm',normed_grid_dat[0,0,0:10])

    #one_data = tf.transpose(one_data, perm=[2,0,1])

    #normed_one_data = one_data/tf.math.reduce_max(one_data)

    num_reps_number = ((3*time_crop)//tf.shape(z1)[0])

    normed_grid_dat = tf.transpose(normed_grid_dat, perm=[2,0,1])

    #int32_num_reps = tf.cast(num_reps_number,tf.int32)

    to_tile = [num_reps_number,1,1]

    rep_normed_grid_dat  = tf.tile(normed_grid_dat,to_tile) # in there for random starting frame

    #normed_grid_dat = tf.transpose(normed_grid_dat, perm=[1,2,0])

    rand_start_frame = tf.experimental.numpy.random.randint(0,high = time_crop - 1) 
    time_crop_rand_start = rep_normed_grid_dat[rand_start_frame:rand_start_frame + time_crop, :, :]

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
#print('ORIGINAL','max y',tf.math.reduce_max(image),'min y',tf.math.reduce_min(image),'type',image.dtype,'shape',tf.shape(image))
#print('image old way',np.shape(image))

#loading data in from .mat file
#image = load_data(file='/media/sf_ML_work/paper_data_mat_files/SAXdataAll.mat')
#print('image',np.shape(image))

#interpolate data to 40 frames with specified accleration factor and start off at random initial temporal frame
time_crop = 40
#image = interpolate_in_time(image, accel = 3, time_crop=time_crop)

naugment=1 #it seems that this determines the range that augment_counter goes up to. 
stopmean=0
#for i in range(naugment): 
start=time.time()
#mask2 and image are the same in this case (both are y) 
x,y=wrapper_augment(image,image,gpu=0,time_crop=time_crop,regsnr=100,deterministic=1,det_counter=10)
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


