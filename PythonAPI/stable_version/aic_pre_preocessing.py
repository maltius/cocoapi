# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:50:19 2020

@author: altius
"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/'
jpeg_path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902'
file_name = 'keypoint_train_annotations_20170902.json'
path_save = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons17/'

labels_raw = json.loads(open(os.path.join(path,file_name)).read()) 
labels=np.zeros((750000,17,3))

c=0;
file_name=list()
for ind in range(len(labels_raw)):
    data_peak=labels_raw[ind]
    I = cv2.imread(os.path.join(jpeg_path,data_peak['image_id'])+'.jpg')
    if ind%100==0:
        print(ind)
    for j in range(0,15):
        try:
            next_one='human'+str(j+1)
            labels_temp=data_peak['keypoint_annotations'][next_one]
            labels_arr=np.copy(np.array(labels_temp).reshape(14,3))
            labels_arr[:,2]=3-labels_arr[:,2]
            coord=data_peak['human_annotations'][next_one]
            # plt.imshow(I)
            # img_part = I[coord[1]:coord[3],coord[0]:coord[2],:]
            # img_part = I[coord[1]:coord[3],coord[0]:coord[2],:]
            wid1=coord[3]-coord[1]
            len1=coord[2]-coord[0]
            
            # skeleton=labels_arr
            # for u in range(14):
            #     cv2.circle(I, center=tuple(skeleton[u,0:2]), radius=10, color=(255, 0, 0), thickness=5);
            # plt.imshow(I)
            
            img_part = I[max(1,coord[1]-int(wid1/3)):min(coord[3]+int(wid1/3),I.shape[0]),max(1,coord[0]-int(len1/1.5)):min(coord[2]+int(len1/1.5),I.shape[1]),:]
            # plt.imshow(img_part)
            image_ind='img_'+str(ind).zfill(8)+'_'+str(j).zfill(2)
            
            nonzeroind = np.nonzero(labels_arr) 
            save_cond=1
            imres=0
            
            # if nonzeroind[0].shape[0]>25.5 and I is not None:
            
            if min((wid1,len1))>30 and nonzeroind[0].shape[0]>41.5:   #labels_arr[0,0]*labels_arr[3,0]*labels_arr[6,0]*labels_arr[9,0]>0:
                file_name.append(image_ind+'.jpeg')
                
                
                
                labels_temp=data_peak['keypoint_annotations'][next_one]
                labels_arr=np.copy(np.array(labels_temp).reshape(14,3))
                labels_arr[:,2]=3-labels_arr[:,2]
                for k in range(labels_arr.shape[0]):
                    if labels_arr[k,2]>0:
                        # if 
                        labels_arr[k,0]=labels_arr[k,0]-max(1,coord[0]-int(len1/1.5))
                        if labels_arr[k,0]<-0.5:
                            labels_arr[k,0]=0
                        labels_arr[k,1]=labels_arr[k,1]-max(1,coord[1]-int(wid1/3))
                        if labels_arr[k,1]<-0.5:
                            labels_arr[k,1]=0

                        
                labels[c,0:14,:]=labels_arr
                cv2.imwrite(os.path.join(path_save,image_ind)+'.jpeg',img_part)
                c=c+1
                
        except:
            pass
            

if False:
    new_labels1=labels[0:c,:,:]
    
    np.save('data_configs_aic/aic_pre_cropped_17.npy',new_labels1)
    new_file_name=np.array(file_name)
    np.save('data_configs_aic/files_aic_pre_cropped_17.npy',np.array(new_file_name))  

            
if False: 
    plt.imshow(I)
    plt.imshow(img_part)
    img=np.copy(img_part)
          
    skeleton=labels_arr
    for u in range(14):
        cv2.circle(img, center=tuple(skeleton[u,0:2]), radius=10, color=(255, 0, 0), thickness=5);
    plt.imshow(img)

    
    