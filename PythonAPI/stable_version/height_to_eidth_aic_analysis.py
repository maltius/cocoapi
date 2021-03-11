import argparse
import logging
import sys
import time
import os
import json


import cv2
import numpy as np




paths='/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'


path='/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
path1='/mnt/sda1/downloads/action/'
jfiles = [f for f in os.listdir(path) if f.endswith('.jpg')]
filesnames=list([])
    
path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/'
jpeg_path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
file_name = 'keypoint_train_annotations_20170902.json'
    # path_save = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons_17_single/'

labels_raw = json.loads(open(os.path.join(path,file_name)).read())
   
height=list()
width=list()
ratio=list()

for t in range(len(labels_raw)):
    
    for j in range(len(labels_raw[t]['human_annotations'].keys())):
        try:
            label=labels_raw[t]['keypoint_annotations']['human'+str(j+1)]
            labels=np.array(label).reshape(14,3)
            labels=labels[:,0:2]
            if t%1000==0:
                print(str(t)+'  number of files is processed out of '+str(len(jfiles)))
            if labels[12,0]*labels[13,0]*labels[6,0]*labels[9,0]>0 and np.linalg.norm(labels[6,:]-labels[9,:])>5:
                mid_down=(labels[6,:]+labels[9,:])/2
                height.append(np.linalg.norm(mid_down-labels[12,:]))
                width.append(np.linalg.norm(labels[6,:]-labels[9,:]))
                ratio.append(height[-1]/width[-1])
                
        except:
            pass
                
        
import matplotlib.pyplot as plt

# rng = np.random.RandomState(10)  # deterministic random data

# a = np.hstack((rng.normal(size=1000),

#                rng.normal(loc=5, scale=2, size=1000)))

_ = plt.hist(np.array(ratio), bins=np.arange(12) ) # arguments are passed to np.histogram

plt.title("Histogram with 'auto' bins")

plt.show()


                