#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:26:00 2021

@author: altius
"""

import numpy as np
import cv2
import os

pat='/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons_17_single/'


all_labels=np.load('all_labels.npy')
all_coord=np.load('all_coord.npy')
X_train_filenames = np.load('new_names.npy')

jfiles = [f for f in os.listdir(pat) if f.endswith('.jpeg')]
# jfiles = srte

for i in range(len(jfiles)):
    # try:
    if i%500==0:
        print('processing batch  '+str(i))

    if i>64999:
        # FileName = '/mnt/sda1/downloads/cocoapi-master/PythonAPI'+X_train_filenames[i][23:]
        Filesave = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/resized128/aic_persons_17_single/'#+X_train_filenames[i][23:]
        # FileName = batch_x[i]
    
        I = cv2.imread(pat+jfiles[i])
        II=cv2.resize(I,(128,128))
        cv2.imwrite(Filesave+jfiles[i], II)
        
    # except:
    #     print(FileName)
        
        
        
        