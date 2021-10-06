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

import pdb
import h5py

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


def wrapper_augment(imagex, imagey,
                    gpu=1,maxrot=45.0,time_axis=2,time_crop=None,min_motion=0, max_motion=0,
                    central_crop=128, grid_size=[192,192],regsnr=8,deterministic=0,det_counter=10,
                    trajfile='/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5',
                    ):

  seed = rng.make_seeds(2)[0]
  if deterministic==1:
      global deterministic_counter
      seed=precomputedseeds[deterministic_counter*2:deterministic_counter*2+2]
      deterministic_counter=(deterministic_counter+1)%det_counter
      
  if gpu==1:
      with tf.device('/gpu:0'):
          if max_motion>0:
              x, y = training_augmentation_flow_withmotion((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,trajfile=trajfile,min_motion_ampli=min_motion,max_motion_ampli=max_motion)
          else:
              x, y = training_augmentation_flow((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,trajfile=trajfile)
  else:
      if max_motion>0:
              x, y = training_augmentation_flow_withmotion((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,trajfile=trajfile,min_motion_ampli=min_motion,max_motion_ampli=max_motion)
      else:
              x, y = training_augmentation_flow((imagey, imagex), seed,maxrot=maxrot,time_axis=time_axis,time_crop=time_crop,central_crop=central_crop,grid_size=grid_size,regsnr=regsnr,trajfile=trajfile)

  return x, y

#augmentations applied to x and y, then x is undersampled at the end
def training_augmentation_flow(image_label,seed,maxrot=45.0,time_axis=2,time_crop=None,central_crop=128,grid_size=[192,192],regsnr=8,trajfile='/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5'):

    maxrot=maxrot/180*tf.constant(math.pi) #Convert to radians
    normseed=tf.cast(seed/9223372036854775807,tf.float32) #[-1;1]
    traj,dcw=loadtrajectory(trajfile)
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


#image_label is (image_y, image_x) which are both y in this case
def training_augmentation_flow_withmotion(image_label,seed,maxrot=45.0,time_axis=2,time_crop=None,central_crop=128,grid_size=[192,192],regsnr=8,min_motion_ampli=0,max_motion_ampli=0,trajfile='/home/oj20/UCLjob/Project2/resources/traj_SpiralPerturbedOJ_section1.h5'):

    normseed=tf.cast(seed/9223372036854775807,tf.float32) #random numbers in this range [-1;1]
    #print('normseed',normseed) #
    traj,dcw=loadtrajectory(trajfile)
    #print('traj shape is',np.shape(traj))
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
    cpx=tf.roll(cpx, shift=tf.cast(normseed[0]*image_size_float[time_axis],tf.int32) , axis=time_axis) #this may be the key line
    #time_crop
    if time_crop is not None:
        #works for time_axis=2
        cpx=cpx[:,:,:time_crop,...]
    cpx_size=tf.shape(cpx)
    
    cpx=tf.expand_dims(cpx, axis=0)
    cpx=tf.concat((tf.math.real(cpx),tf.math.imag(cpx)),axis=0)
    
    #MOTION
    #ApplyMotion on 50% of data (exercise vs rest)
    cpx=tf.transpose(cpx,[3,1,2,0]) #what is cpx?
    global augment_counter
    #random motion amplitude inbetween specified limits
    motion_ampli = tf.random.uniform([1],minval=min_motion_ampli,maxval=max_motion_ampli,dtype=tf.dtypes.float32,seed=None,name=None)
    #print('motion_ampli',motion_ampli)
    print('augment_counter',augment_counter)
    if augment_counter%2==1: #if it is even
        
        displacement=[]
        for idxdisp in range(time_crop): #40
            print('idxdisp',idxdisp)
            if idxdisp%40==0:               
                #transform=tf.random.stateless_uniform((2,1), seed+idxdisp, minval=-motion_ampli, maxval=motion_ampli) #shift in 2 dimensions
                transform=tf.random.stateless_uniform((1,1), seed+idxdisp, minval=-motion_ampli, maxval=motion_ampli) #shift in 1 dimension
                print('transform',transform)
            sin_point = 20*float(transform)*np.sin((idxdisp/time_crop)*2*np.pi)
            #sin_point = 20*np.sin((idxdisp/time_crop)*2*np.pi)
            #this gives you a random number between the min and max amplitudes
            #print('seed',seed)
            #print('transform',transform)
            if idxdisp==0:
                displacement.append(tf.cast([[0],[0]],tf.float32)) #this is to get the initial zeros at the start
                #displacement.append(0)
            # else:
                #displacement.append(transform+displacement[idxdisp-1]) #add on the same random number 20 times
            #print('sin_point in',sin_point)
            displacement.append(tf.cast([[sin_point],[sin_point]],tf.float32))
            #displacement.append(sin_point)
            #print('displacement in',displacement)
        #print('sin_point',sin_point)
        displacement=tf.stack(displacement) #shape(20,2,1) values from 0 to -20.
        #print('displacement',displacement)
        transform=tf.squeeze(displacement) #shape(20,2) values from 0 to -20
        #print('squeezed transform',transform)
        #cpx size (20,192,192,2). this translates the cpx by the transform
        cpx=tfa.image.translate(cpx,transform[:-1,:],'bilinear' ) #image:cpx. transaltion:transform. interpolation mode: bilinear
        
    #ROTATION
    maxrot=maxrot/180*tf.constant(math.pi) #Convert to radians.
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
    
    #NUFFT  First a forward transform and then a backward transform ONLY DOING THIS TO x
    #print('traj with trajstart shape is',np.shape(traj[trajstart:(trajstart+cpx_size[time_axis]),...]))
    kspace = tfft.nufft(x, traj[trajstart:(trajstart+cpx_size[time_axis]),...],transform_type='type_2', fft_direction='forward') 
    #print('kspace shape',np.shape(kspace))
    #print('dcw shape',np.shape(dcw))
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
trajfile='/media/sf_ML_work/trajectory_files/traj_tGAOJ_13.h5'
fdata='/media/sf_ML_work/mapped_docker_files/ml/data/yonly/a_few_SAX_40_rest_and_trans_aug_y_only/cache/'

dataplot='train_00001'
filename=fdata + dataplot + '.h5'

ori= h5py.File(filename, 'r')
image=ori['y']
mask2=ori['y']
#image=(img[0,:,:,:]+1j*img[1,:,:,:])/np.max(img)
#ys=np.concatenate((tf.abs(image[32:160,32:160,1])+mask2[32:160,32:160,1],(tf.math.angle(image[32:160,32:160,1])+np.pi)/(2*np.pi)),axis=1)
naugment=2 #it seems that this determines the range that augment_counter goes up to. Only works if left at 6.
stopmean=0
for i in range(naugment): #6
    start=time.time()
    #mask2 and image are the same in this case (both are y) 
    x,y=wrapper_augment(mask2,image,min_motion=1,max_motion=5,time_crop=40,regsnr=100,deterministic=1,det_counter=10,trajfile=trajfile)
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
print('imgx',np.shape(imgx),'max imgx',np.amax(imgx))
PlotUtils.plotVid(imgx,axis=0,vmax=1)





