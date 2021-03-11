#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:05:55 2021

@author: altius
"""

import os
os.chdir('/mnt/sda1/downloads/BlazePose-tensorflow-master/')


import platform
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from config import num_joints, batch_size, gaussian_sigma, gpu_dynamic_memory
import cv2



from config import total_epoch, train_mode

from model import BlazePose


model=BlazePose()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_all3_kp_LR_001_13/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=14))


path='/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/main/'


jfiles = [f for f in os.listdir(path) if f.endswith('.jpg')]



train_data = np.zeros([len(jfiles), 256, 256, 3])


c=0
for i in range(len(jfiles)):
    try:
        if i%10==0:
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


y = np.zeros((c,num_joints, 3)).astype(np.uint8)
test_sam1=train_data[0:c]
keyp=num_joints
y = model(test_sam1).numpy().astype(np.uint8)

for t in range(c):
       tt=0
       skeleton = y[t]
       img = train_data[t].astype(np.uint8)
       for i in range(keyp):
           cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(255, 0, 0), thickness=2)
       cv2.imwrite(path+"./results/test"+str(t).zfill(5)+".jpg", img)

        
            

