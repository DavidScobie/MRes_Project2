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
import scipy.io as sio
import pdb
import h5py

rng = tf.random.Generator.from_seed(123, alg='philox')
rng2 = tf.random.Generator.from_seed(1234, alg='philox')  
precomputedseeds=rng2.make_seeds(1000)[0]
global augment_counter
augment_counter=0
deterministic_counter=0 #reset every det_counter iterations (max 500) -> det_counter set at validation set size to have the same transform at each validation step.

# def loadtrajectory(trajfile):
#     # global traj
#     # global dcw
#     # global augment_counter
#     with h5py.File(trajfile, 'r') as f:
#             traj = f['traj'][...]/tf.reduce_max(f['traj'][...])*tf.constant(math.pi)
#             dcw = f['dcw'][...]
#     #augment_counter=1       
#     traj=tf.reshape(traj,(traj.shape[0],traj.shape[1]*traj.shape[2],traj.shape[3]))
#     dcw=tf.cast(tf.reshape(dcw,(dcw.shape[0],dcw.shape[1]*dcw.shape[2])),dtype=tf.complex64)
#     return traj,dcw
#loadtrajectory('/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5')


def wrapper_augment(imagex, imagey,
                    gpu=1,maxrot=45.0,time_axis=2,time_crop=None,motion=0,
                    central_crop=128, grid_size=[192,192],regsnr=8,deterministic=0,det_counter=10,AngleFile=None
                    ):

  seed = rng.make_seeds(2)[0]
  if deterministic==1:
      global deterministic_counter
      seed=precomputedseeds[deterministic_counter*2:deterministic_counter*2+2]
      deterministic_counter=(deterministic_counter+1)%det_counter
      
  if gpu==1:
      with tf.device('/gpu:0'):
          if motion>0:
              x, y = training_augmentation_flow_withmotion((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,AngleFile=AngleFile,motion_ampli=motion)
          else:
              x, y = training_augmentation_flow((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr)
  else:
      if motion>0:
              x, y = training_augmentation_flow_withmotion((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,AngleFile=AngleFile,motion_ampli=motion)
      else:
              x, y = training_augmentation_flow((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr)

  return x, y

def training_augmentation_flow(image_label,seed,maxrot=45.0,time_axis=2,time_crop=None,central_crop=128,grid_size=[192,192],regsnr=8,trajfile='/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5'):

    maxrot=maxrot/180*tf.constant(math.pi) #Convert to radians
    normseed=tf.cast(seed/9223372036854775807,tf.float32) #[-1;1]
    #traj,dcw=loadtrajectory(trajfile)
    y,x= image_label
    
    y=tf.transpose(y,perm=(0,1,2))
    y=tf.cast(y,tf.complex64)
    
    image_size=tf.shape(y)
    image_size_float=tf.cast(tf.shape(y),tf.float32)
    mag=tf.math.abs(y)#/tf.reduce_max(tf.math.abs(y),axis=(0,1),keepdims=True) #for frameperframe norm
    phs=tf.math.angle(y)+normseed[0]*tf.constant(0.5*math.pi)
    
    #Contrast Change on magnitude
    mag=tf.image.stateless_random_contrast(mag,lower=0.5,upper=1.5 , seed=seed)
    
    cpx=tf.cast(mag,tf.complex64)*tf.exp(1j*tf.cast(phs,tf.complex64))
    del mag,phs
    
    #Flip/Roll
    cpx=tf.image.stateless_random_flip_left_right(cpx, seed=seed)
    cpx=tf.image.stateless_random_flip_up_down(cpx, seed=seed)
    cpx=tf.roll(cpx, shift=tf.cast(normseed[0]*image_size_float[time_axis],tf.int32) , axis=time_axis)
    #time_crop
    if time_crop is not None:
        #works for time_axis=2
        cpx=cpx[:,:,:time_crop,...]
    cpx_size=tf.shape(cpx)
    #ROTATION
    cpx=tf.expand_dims(cpx, axis=0)
    cpx=tf.concat((tf.math.real(cpx),tf.math.imag(cpx)),axis=0)
    cpx=tfa.image.rotate(cpx, angles=(normseed[0])*maxrot+maxrot,interpolation='bilinear')
    #Crop to original size (before rotation)
    cpx=cpx[:,cpx_size[0]//2-image_size[0]//2:(cpx_size[0]//2-image_size[0]//2+image_size[0]),
            cpx_size[1]//2-image_size[1]//2:(cpx_size[1]//2-image_size[1]//2+image_size[0]),...]
    
    """
    X is overwritten
    """
    
    #Add White gaussian noise to x branch only
    # x=awgn(cpx,regsnr,seed) #previously uncommented (if you want noise)
    #cpx=awgn(cpx,regsnr)
    x = cpx  # if you dont want any noise
    
    cpx=tf.complex(cpx[0,...],cpx[1,...])
    cpx=tf.transpose(cpx,perm=(2,0,1))
    
    x=tf.complex(x[0,...],x[1,...])
    x=tf.transpose(x,perm=(2,0,1))
    
    #Random Trajectory start point
    trajstart=tf.cast((normseed[0]+1)/2*tf.cast(tf.shape(traj)[0]-cpx_size[time_axis],tf.float32),tf.int32)
    
    #NUFFT
    kspace = tfft.nufft(x, traj[trajstart:(trajstart+cpx_size[time_axis]),...],transform_type='type_2', fft_direction='forward')
    x = tfft.nufft(kspace*dcw[trajstart:(trajstart+cpx_size[time_axis]),...], traj[trajstart:(trajstart+cpx_size[time_axis]),...], grid_shape=grid_size, transform_type='type_1', fft_direction='backward')
    

    
    """
    Crop
    """
    cpx=cpx[:,cpx_size[0]//2-central_crop//2:(cpx_size[0]//2-central_crop//2+central_crop),
            cpx_size[1]//2-central_crop//2:(cpx_size[1]//2-central_crop//2+central_crop)]
    x=x[:,cpx_size[0]//2-central_crop//2:(cpx_size[0]//2-central_crop//2+central_crop),
            cpx_size[1]//2-central_crop//2:(cpx_size[1]//2-central_crop//2+central_crop)]

    
    cpx=tf.expand_dims(cpx,axis=-1)
    x=tf.expand_dims(x,axis=-1)
    
    cpx=cpx/tf.cast(tf.reduce_max(tf.abs(cpx)),dtype=cpx.dtype)
    x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)
    
    y=tf.math.abs(cpx)
    x=tf.math.abs(x)
    
    x=tf.ensure_shape(x,(None,central_crop,central_crop,1))
    y=tf.ensure_shape(y,(None,central_crop,central_crop,1))
    # y=tf.concat((tf.math.real(cpx),tf.math.imag(cpx)),axis=-1)
    # x=tf.concat((tf.math.real(x),tf.math.imag(x)),axis=-1)
    
    # x=tf.ensure_shape(x,(None,central_crop,central_crop,2))
    # y=tf.ensure_shape(y,(None,central_crop,central_crop,2))
    
    return x,y

def awgn(data, regsnr,seed): 
    """
    Add White Gaussian noise to reach target snr
    """
    sigpower = tf.reduce_mean(tf.abs(data) ** 2)
    noisepower = sigpower / (10 ** (regsnr / 10))
    noise = tf.random.stateless_normal(tf.shape(data), seed,0, tf.math.sqrt(noisepower))
    #noise = tf.random.normal( tf.shape(data),0, tf.math.sqrt(noisepower))
    data += noise
    return data

# @tf.function
# def motion_aug(cpx,seed,displacement=None,interp='bilinear',motion_proba=0): 
#     normseed=tf.cast(seed/9223372036854775807,tf.float32) #[-1;1]
#     if (normseed[1]+1)/2<motion_proba:
#         if displacement is None:
#             displacement=[]
#             for idxdisp in range(tf.shape(cpx)[0]):
#                 print(seed[0])
#                 transform=tf.random.normal( (2,1),0,1,seed=seed[0])
#                 if idxdisp==0:
#                     displacement.append(tf.cast([[0], [0]],tf.float32))
#                 else:
#                     displacement.append(transform+displacement[idxdisp-1])
#             displacement=tf.stack(displacement)
#             displacement=tf.squeeze(displacement)
#         cpx=tfa.image.translate(cpx,displacement,interpolation=interp )
    
#     return cpx, displacement

def create_trajectory(AngleFile='/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5'):
    contents = sio.loadmat(AngleFile)
    radialAngles = contents['radialAngles']
    print('radialAngles',np.shape(radialAngles))

def training_augmentation_flow_withmotion(image_label,seed,maxrot=45.0,time_axis=2,time_crop=None,central_crop=128,grid_size=[192,192],regsnr=8,motion_ampli=0.5,AngleFile=None):
    
    traj = create_trajectory(AngleFile=AngleFile)
    
    maxrot=maxrot/180*tf.constant(math.pi) #Convert to radians
    normseed=tf.cast(seed/9223372036854775807,tf.float32) #[-1;1]
    #traj,dcw=loadtrajectory(trajfile)
    y,x= image_label
    
    y=tf.transpose(y,perm=(1,2,0))
    
    y=tf.cast(y,tf.complex64)
    
    image_size=tf.shape(y)
    image_size_float=tf.cast(tf.shape(y),tf.float32)
    mag=tf.math.abs(y)#/tf.reduce_max(tf.math.abs(y),axis=(0,1),keepdims=True) #for frameperframe norm
    phs=tf.math.angle(y)+normseed[0]*tf.constant(0.5*math.pi)
    
    #Contrast Change on magnitude
    mag=tf.image.stateless_random_contrast(mag,lower=0.5,upper=1.5 , seed=seed)
    
    cpx=tf.cast(mag,tf.complex64)*tf.exp(1j*tf.cast(phs,tf.complex64))
    del mag,phs
    
    #Flip/Roll
    cpx=tf.image.stateless_random_flip_left_right(cpx, seed=seed)
    cpx=tf.image.stateless_random_flip_up_down(cpx, seed=seed)
    cpx=tf.roll(cpx, shift=tf.cast(normseed[0]*image_size_float[time_axis],tf.int32) , axis=time_axis)
    #time_crop
    if time_crop is not None:
        #works for time_axis=2
        cpx=cpx[:,:,:time_crop,...]
    cpx_size=tf.shape(cpx)
    
    cpx=tf.expand_dims(cpx, axis=0)
    cpx=tf.concat((tf.math.real(cpx),tf.math.imag(cpx)),axis=0)
    
    #MOTION
    #ApplyMotion on 50% of data (exercise vs rest)
    cpx=tf.transpose(cpx,[3,1,2,0])
    global augment_counter
    if augment_counter%2==1:
        displacement=[]
        for idxdisp in range(time_crop):
            if idxdisp%10==0:
                transform=tf.random.stateless_uniform((2,1), seed+idxdisp, minval=-motion_ampli, maxval=motion_ampli)
            if idxdisp==0:
                displacement.append(tf.cast([[0], [0]],tf.float32))
            else:
                displacement.append(transform+displacement[idxdisp-1])
        displacement=tf.stack(displacement)
        transform=tf.squeeze(displacement)
        cpx=tfa.image.translate(cpx,transform,'bilinear' )
        
    #ROTATION
    cpx=tfa.image.rotate(cpx, angles=(normseed[0])*maxrot+maxrot,interpolation='bilinear')
    #Crop to original size (before rotation)
    cpx=cpx[:,cpx_size[0]//2-image_size[0]//2:(cpx_size[0]//2-image_size[0]//2+image_size[0]),
            cpx_size[1]//2-image_size[1]//2:(cpx_size[1]//2-image_size[1]//2+image_size[0]),...]
    cpx=tf.transpose(cpx,[3,1,2,0])
    augment_counter+=1
    #print('Stop1',time.time()-start)
    """
    X Overwritten
    """
    
    #Add White gaussian noise to x branch only
    x=awgn(cpx,regsnr,seed)
    
    cpx=tf.complex(cpx[0,...],cpx[1,...])
    cpx=tf.transpose(cpx,perm=(2,0,1))
    
    x=tf.complex(x[0,...],x[1,...])
    x=tf.transpose(x,perm=(2,0,1))
    #Random Trajectory start point
    trajstart=tf.cast((normseed[0]+1)/2*tf.cast(tf.shape(traj)[0]-cpx_size[time_axis],tf.float32),tf.int32)
    
    #NUFFT
    kspace = tfft.nufft(x, traj[trajstart:(trajstart+cpx_size[time_axis]),...],transform_type='type_2', fft_direction='forward')
    x = tfft.nufft(kspace*dcw[trajstart:(trajstart+cpx_size[time_axis]),...], traj[trajstart:(trajstart+cpx_size[time_axis]),...], grid_shape=grid_size, transform_type='type_1', fft_direction='backward')

    """
    Crop
    """

    cpx=cpx[:,cpx_size[0]//2-central_crop//2:(cpx_size[0]//2-central_crop//2+central_crop),
            cpx_size[1]//2-central_crop//2:(cpx_size[1]//2-central_crop//2+central_crop)]
    x=x[:,cpx_size[0]//2-central_crop//2:(cpx_size[0]//2-central_crop//2+central_crop),
            cpx_size[1]//2-central_crop//2:(cpx_size[1]//2-central_crop//2+central_crop)]

    
    cpx=tf.expand_dims(cpx,axis=-1)
    x=tf.expand_dims(x,axis=-1)
    #label=tf.expand_dims(label,axis=-1)
    
    cpx=cpx/tf.cast(tf.reduce_max(tf.abs(cpx)),dtype=cpx.dtype)
    x=x/tf.cast(tf.reduce_max(tf.abs(x)),dtype=x.dtype)
    #label[label>0]=label[label>0]/tf.reduce_max(label)
    
    # y=tf.concat((tf.math.real(cpx),tf.math.imag(cpx)),axis=-1)
    # x=tf.concat((tf.math.real(x),tf.math.imag(x)),axis=-1)
    
    # x=tf.ensure_shape(x,(None,central_crop,central_crop,2))
    # y=tf.ensure_shape(y,(None,central_crop,central_crop,2))
    
    y=tf.math.abs(cpx)
    x=tf.math.abs(x)
    
    x=tf.ensure_shape(x,(None,central_crop,central_crop,1))
    y=tf.ensure_shape(y,(None,central_crop,central_crop,1))
    
    return x,y



#Quick Test
import h5py
import numpy as np
import sys
# sys.path.insert(0, '/home/oj20/UCLjob/PythonCode/Utils/')
sys.path.insert(0, '/sf_ML_work/read_DICOMS/')

import PlotUtils
import time
#trajfile='/media/sf_ML_work/trajectory_files/traj_tGAOJ_13.h5'
fdata='/media/sf_ML_work/mapped_docker_files/ml/data/yonly/a_few_SAX_40_rest_and_trans_aug_y_only/cache/'

dataplot='train_00001'
filename=fdata + dataplot + '.h5'

ori= h5py.File(filename, 'r')
image=ori['y']
mask2=ori['y']
#image=(img[0,:,:,:]+1j*img[1,:,:,:])/np.max(img)
#ys=np.concatenate((tf.abs(image[32:160,32:160,1])+mask2[32:160,32:160,1],(tf.math.angle(image[32:160,32:160,1])+np.pi)/(2*np.pi)),axis=1)
naugment=6
stopmean=0
for i in range(naugment):
    start=time.time()
    x,y=wrapper_augment(mask2,image,motion=0.9,time_crop=20,regsnr=100,deterministic=1,det_counter=10,AngleFile='/media/sf_ML_work/BART/meas_MID00576_FID48987_ex3_stack_raw_and_ang.mat')
    stop=time.time()-start
    stopmean=stopmean+stop
    #pdb.run('x,y=wrapper_augment(image,mask2)')

    temp=np.concatenate((tf.abs(x[-1,...]), tf.abs(y[-1,...])),axis=1)
    if i==0:
        ys=temp
    else:
        
        ys=np.concatenate((ys,temp),axis=0)
    
stopmean=stopmean/naugment
print('Mean time:',stopmean,'\n Last time:',stop)
# PlotUtils.plotXd(ys,vmax=1,vmin=0)
imgx=np.concatenate((x,y),axis=1)
PlotUtils.plotVid(imgx,axis=0,vmax=1)





