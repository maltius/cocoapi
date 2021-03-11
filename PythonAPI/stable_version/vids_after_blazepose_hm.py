#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:05:55 2021

@author: altius
"""

import os
os.chdir('/mnt/sda1/downloads/BlazePose-tensorflow-master/')
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

import platform
import numpy as np

from scipy.io import loadmat
from config import num_joints, batch_size, gaussian_sigma, gpu_dynamic_memory
import cv2
import glob
import matplotlib.pyplot as plt
import time



from config import total_epoch, train_mode

from model import BlazePose

def bone(skeleton,img,keyp):
    if skeleton.shape[1]==3:
        skeleton=skeleton[:,0:2]
    bonepairs=[(0,1),(1,2),(0,12),(3,4),(4,5),(12,3),(3,9),(9,10),(10,11),(0,6),(6,7),(7,8)]
    bone_pairs1=list()
 
    for bones in bonepairs:
        if skeleton[bones[1],0]>0 and skeleton[bones[0],0]>0:
            cv2.line(img, (skeleton[bones[1],0], skeleton[bones[1],1]), (skeleton[bones[0],0], skeleton[bones[0],1]), (255, 0, 0), thickness=2)
    return img


def rot_back(im_rot_sh,image_sh, xy, a):
    # im_rot = ndimage.rotate(image,angle) 
    org_center = (np.array(image_sh[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot_sh[:2][::-1])-1)/2.
    org = xy-org_center
    # a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center    

def revert_back(labels,rot_info):
    
    
    labels=labels[:,0:2]
    img_shape=rot_info['final_size']
    labels[:, 0] *= (img_shape[1] / 256)
    labels[:, 1] *= (img_shape[0] / 256)
    
    add20=rot_info['add20']
    add21=rot_info['add21']

    for u in range(labels.shape[0]):
    # labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
        labels[u,0]=labels[u,0]+add20
        labels[u,1]=labels[u,1]+add21
        
    filler_rot_sh=rot_info['rot_im_size']
    filler_sh=rot_info['im_size']
    angle=rot_info['angle']

    for u in range(labels.shape[0]):
        labels[u,:]=rot_back(filler_rot_sh,filler_sh,labels[u,:],-angle)


    
    stpoint1=rot_info['add10']
    stpoint0=rot_info['add11']    
    
    for u in range(labels.shape[0]):
        labels[u,0]=labels[u,0]+stpoint1
        labels[u,1]=labels[u,1]+stpoint0
        
        
    stpoint1=rot_info['add30']
    stpoint0=rot_info['add31']    
    
    for u in range(labels.shape[0]):
        labels[u,0]=labels[u,0]+stpoint1
        labels[u,1]=labels[u,1]+stpoint0

    

        
    return labels
    



model=BlazePose()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_all3_hm_LR_001_13/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=8))


path='/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/resized/'
path1='/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/'

jfiles1 = [f for f in os.listdir(path) if f.endswith('.jpg')]

jfiles=sorted(jfiles1)

files = glob.glob('/mnt/sda1/downloads/BlazePose-tensorflow-master/results_hm/*.jpg')
for f in files:
    os.remove(f)


train_data = np.zeros([len(jfiles), 256, 256, 3])


c=0
for i in range(len(jfiles)):
    try:
        if i%100==0:
            print(i)
        # FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
        FileName = os.path.join(path,jfiles[i])
        # FileName=FileName[0:45]+"aligned_"+FileName[45:]
        ii=cv2.imread(FileName)
        if min(ii.shape[0:2])>30:
            img = tf.io.read_file(FileName)
            img = tf.image.decode_image(img)
            img_shape = img.shape
            # Attention here img_shape[0] is height and [1] is width
            train_data[c] = img
            # generate heatmap set
            # for j in range(num_joints):
            #     _joint = (train_label[c, j, 0:2] // 2).astype(np.uint16)
            #     # print(_joint)
            #     train_heatmap_set[c, :, :, j] = getGaussianMap(joint = _joint, heat_size = 128, sigma = gaussian_sigma)
            c=c+1
            
    except:
        pass


y = np.zeros((c,256,256,num_joints)).astype(np.uint8)
test_sam1=train_data[0:c]
keyp=num_joints

a=time.time()
y = model.predict(test_sam1)
b=time.time()
print(b-a)

ex_labels=np.zeros((y.shape[0],y.shape[3],3))
for p in range(y.shape[0]):   
    for q in range(y.shape[3]):
        try_var=(y[p,:,:,q])
        # res=try_var.argmax(axis=(1))
        ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
        ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
            
yy=ex_labels
for t in range(c):
    skeleton = yy[t].astype(int)
    img = train_data[t].astype(np.uint8)
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
    cv2.imwrite(path1+"results_hm/test"+str(t).zfill(5)+".jpg", img)
    
    
tot_rot_info=np.load('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/rot_info.npy',allow_pickle=True).item()



yy=np.copy(ex_labels)
for t in range(c):
    skeleton = revert_back(yy[t,:,0:2],tot_rot_info['sample'+str(t).zfill(5)]).astype(int)
    img = cv2.imread(path1+'/main/'+'sample'+str(t).zfill(5)+'.jpg')
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=5, color=(0, 255, 0), thickness=5)
        img1=bone(skeleton,img,13)
    cv2.imwrite(path1+"results_hm_main/test"+str(t).zfill(5)+".jpg", img)
    
# rot_info=tot_rot_info['sample'+str(t).zfill(5)]


# labels=np.copy(yy[0,:])




# def rot(im_rot,image, xy, a):
#     # im_rot = ndimage.rotate(image,angle) 
#     org_center = (np.array(image.shape[:2][::-1])-1)/2.
#     rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
#     org = xy-org_center
#     # a = np.deg2rad(angle)
#     new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
#             -org[0]*np.sin(a) + org[1]*np.cos(a) ])
#     return new+rot_center    

# def align_im(img,labels):
    

    
#     # if body_vec[0]<0:
#     #     angle=angle+90
#     mid_hip_old=mid_hip

    
#     mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
#     mid_hip=np.array([0.5*(labels[8,0]+labels[11,0]),0.5*(labels[8,1]+labels[11,1])]).astype(int)
    
#     diam=int(np.linalg.norm(mid_hip-mid_sh))
#     final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    
    

    
#     return final,labels,rot_info
            

