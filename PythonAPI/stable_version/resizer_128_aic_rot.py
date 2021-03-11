#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:26:00 2021

@author: altius
"""

import numpy as np
import cv2
import os
import math
import random
from scipy import ndimage
import matplotlib.pyplot as plt


def rot(im_rot,image, xy, a):
    # im_rot = ndimage.rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    # a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center    

  

pat='/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons_17_single/'


all_labels=np.load('all_labels.npy')
all_coord=np.load('all_coord.npy')
X_train_filenames = np.load('new_names.npy')

jfiles = [f for f in os.listdir(pat) if f.endswith('.jpeg')]
# jfiles = srte

rot_ang_deg =[-50,-40,-30,-15,0,15,20,30,45]
rot_ang = [ r*(math.pi/180) for r in rot_ang_deg]

for i in range(len(jfiles)):
    # try:
    if i%500==0:
        print('processing batch  '+str(i))

    if True:
        FileName = '/mnt/sda1/downloads/cocoapi-master/PythonAPI'+X_train_filenames[i][23:]
        # Filesave = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/resized128/aic_persons_17_single/'X_train_filenames[i][23:]
        # FileName = batch_x[i]
    
        I = cv2.imread(FileName)
        labels=np.copy(all_labels[i,0,:,:])
        labels[:,0]*= I.shape[0]
        labels[:,1]*= I.shape[1]
        
        c=0
        while c<6:
            rot_val=random.choice(range(len(rot_ang_deg)))
            
            filler_rot = ndimage.rotate(I,rot_ang_deg[rot_val],reshape=False,order=0)

            for u in range(labels.shape[0]):
                labels[u,:]=rot(filler_rot,I,labels[u,:],rot_ang[rot_val])
            
            plt.imshow(filler_rot)
            # plt.imshow(I)
            img1=np.copy(filler_rot)
            skeleton=labels.astype(int)
            for ii in range(6):
                cv2.circle(img1, center=tuple(skeleton[ii][0:2]), radius=4, color=(0, 255, 0), thickness=4)
            plt.imshow(img1)
            
            
            skeleton=labels.astype(int)
            skeleton1=np.copy(skeleton)
            skeleton1[:,0]=np.copy(skeleton[:,1])
            skeleton1[:,1]=np.copy(skeleton[:,0])
            for ii in range(6):
                cv2.circle(I, center=tuple(skeleton1[ii][0:2]), radius=4, color=(0, 255, 0), thickness=4)
            plt.imshow(I)
                
                
        II=cv2.resize(I,(128,128))
        cv2.imwrite(Filesave+jfiles[i], II)
        
    # except:
    #     print(FileName)
        
        
        
        