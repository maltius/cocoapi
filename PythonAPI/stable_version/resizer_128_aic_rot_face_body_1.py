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
import math


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


particle='BFfaces_and_mids_'
all_labels=np.load('pre_calc/aic_labels_'+particle+'.npy')
all_coord=np.load('pre_calc/aic_coord_'+particle+'.npy')
X_train_filenames = np.load('pre_calc/aic_new_names_'+particle+'.npy')

jfiles = [f for f in os.listdir(pat) if f.endswith('.jpeg')]
# jfiles = srte

rot_ang_deg =[0,15,20,30,45]
rot_ang = [ math.radians(r) for r in rot_ang_deg]

for i in range((len(X_train_filenames))):# range(len(X_train_filenames)):
    # try:
    if i%100==0:
        print('processing batch  '+str(i) + ' / '+str(len(X_train_filenames)))

    if True:
        FileName = '/mnt/sda1/downloads/cocoapi-master/PythonAPI'+X_train_filenames[i][23:]
        Filesave = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/resized128_rot_test2'+X_train_filenames[i][23:]
        # FileName = batch_x[i]
    
        I = cv2.imread(FileName)
        labels_temp = np.copy(all_labels[i,0,:,:])
        labels = np.copy(labels_temp)
        # labels[:,0]= np.copy(labels_temp[:,1])
        # labels[:,1]= np.copy(labels_temp[:,0])
        
        labels[:,0]*= I.shape[1]
        labels[:,1]*= I.shape[0]
        
        
        c=0
        while c<6:
            rot_val=random.choice(range(len(rot_ang_deg)))
            
            filler_rot = ndimage.rotate(I,rot_ang_deg[rot_val],reshape=False,order=0,mode='nearest')

            for u in range(labels.shape[0]):
                labels[u,:]=rot(filler_rot,I,labels[u,:],rot_ang[rot_val])
            
            # plt.imshow(filler_rot)
            # # plt.imshow(I)
            # img1=np.copy(filler_rot)
            # skeleton=labels.astype(int)
            # for ii in range(6):
            #     cv2.circle(img1, center=tuple(skeleton[ii][0:2]), radius=4, color=(255, 0, 0), thickness=4)
            # plt.imshow(img1)
            
            
            # skeleton=labels.astype(int)
            # skeleton1=np.copy(skeleton)
            # # skeleton1[:,0]=np.copy(skeleton[:,1])
            # # skeleton1[:,1]=np.copy(skeleton[:,0])
            # for ii in range(6):
            #     cv2.circle(I, center=tuple(skeleton1[ii][0:2]), radius=4, color=(255, 0, 0), thickness=4)
            # plt.imshow(I)
            
            labels_conv = np.copy(labels)
            labels_conv[:,0] *= 1/I.shape[1]
            labels_conv[:,1] *= 1/I.shape[0]
            
            if sum(sum(labels_conv>0.95))==0 and sum(sum(labels_conv<0.04))==0:
                c=11
                all_labels[i,0,:,:]=labels_conv
                minx=min(labels_conv[:,1])
                maxx=max(labels_conv[:,1])
                miny=min(labels_conv[:,0])
                maxy=max(labels_conv[:,0])
                
                coord = np.array([minx*0.95,miny*0.95,maxx*1.05,maxy*1.05]).astype(np.float32)
                if rot_ang_deg[rot_val]:
                    all_coord[c,:,:]=np.reshape(coord,(1,4)).astype(np.float32)
                II=cv2.resize(filler_rot,(128,128))
                cv2.imwrite(Filesave, II)
            else:
                c += 1
                
            
        if c<6.5:
            II=cv2.resize(I,(128,128))
            cv2.imwrite(Filesave, II)

np.save('pre_calc/test2_rot_all_labels.npy',all_labels)
np.save('pre_calc/test2_rot_all_coord.npy',all_coord)
np.save('pre_calc/test2_rot_new_names.npy',X_train_filenames)
    # except:
    #     print(FileName)
        
        
        
        