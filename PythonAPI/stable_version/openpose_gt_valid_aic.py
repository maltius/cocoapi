#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:35:29 2020

@author: altius
"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

label1 = np.load('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/aic_pre_aligned_humans_coco.npy',allow_pickle=True)
file_names = np.load('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/files_aic_pre_aligned_humans_coco.npy')
p_a='/mnt/sda1/downloads/cocoapi-master/PythonAPI/aligned_action_persons_mids/'

path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/'
jpeg_path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902'
file_name1 = 'keypoint_train_annotations_20170902.json'
path_save = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons_17_single/'

labels_raw = json.loads(open(os.path.join(path,file_name1)).read()) 
labels=np.zeros((500000,17,3))

def aic_pck_amp(est_keys,true_keypoints):
    dist=1000
    torso_diam=np.linalg.norm(true_keypoints[3,0:2] - true_keypoints[9,0:2])
    est_key=est_keys[:,0:2]
    true_keypoint=true_keypoints[:,0:2]
    
    dist_all= np.array([ np.linalg.norm(true_keypoint[x,:] - est_key[x,:]) for x in range(est_key.shape[0])])
    
    
    return np.sum(dist_all<torso_diam/5)

distan=list()
c=0;
file_name=list()
for ind in range(0,len(labels_raw)):
    data_peak=labels_raw[ind]
    I = cv2.imread(os.path.join(jpeg_path,data_peak['image_id'])+'.jpg')
    if ind%100==0:
        print(ind)
    for j in range(len(data_peak['human_annotations'].keys())):
        try:
            if I is not None and len(data_peak['human_annotations'].keys())==1:
                next_one='human'+str(j+1)
                labels_temp=data_peak['keypoint_annotations'][next_one]
                labels_arr=np.copy(np.array(labels_temp).reshape(14,3))
                labels_arr[:,2]=3-labels_arr[:,2]
                coord=data_peak['human_annotations'][next_one]
                
                if ind>0:
                    ind_found=-1
                    for d in range(max(0,ind-115),ind+1):
                        if file_names[d][175:]==(data_peak['image_id']+'.jpg'):
                            ind_found=d
                else:
                    ind_found=0
                
                if ind_found>-1 and np.mean(labels_arr[:,2])>1.9:
                    lab1=label1[ind_found] 
                    
                    body_len=np.zeros((5))
                    for w in range(5):
                        try:
                            temp_label_dic=lab1['human'+str(w+1)]
                            temp_label=np.zeros((18,2))
                            for r in range(18):
                                try:
                                    temp_label[r,0]=temp_label_dic[r][0]
                                    temp_label[r,1]=temp_label_dic[r][1]
                                except:
                                    pass
                            minx=int(min(temp_label[:,1]))
                            maxx=int((max(temp_label[:,1])))
                            miny=int(min(temp_label[:,0]))
                            maxy=int((max(temp_label[:,0])))
                            im_size=max(maxx-minx,maxy-miny)
                            body_len[w]=im_size
                        except:
                            pass
                        
                    max_size=np.argmax(body_len)
                    
                
                    temp_label_dic=lab1['human'+str(max_size+1)]
                    temp_label=np.zeros((18,2))
                    for r in range(18):
                        try:
                            temp_label[r,0]=temp_label_dic[r][0]
                            temp_label[r,1]=temp_label_dic[r][1]
                        except:
                            pass
                    
                    
                    lab2=labels_arr[:,0:2]
                    
                    lab2[12,:]=[0,0]
                    lab3=np.zeros((lab2.shape))
                    for q in range(2,lab3.shape[0]):
                        lab3[q-2,:]=temp_label[q,:]
                    lab3[13,:]=temp_label[1,:]
                        
                    distan.append(aic_pck_amp(lab3,lab2))
        except:
            pass
            
plt.imshow(I)    

skeleton=temp_label
for ii in range(max(skeleton.shape)):
    # cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=2, color=(0, 255, 0), thickness=20)
    cv2.putText(I,str(ii), tuple(skeleton[ii][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),thickness=5)
        
plt.imshow(I)    



