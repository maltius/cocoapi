#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:34:39 2020

@author: altius
"""


from pycocotools.coco import COCO
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import requests
import cv2
import time

# where all images containing "person filtered from coco"
persons_img_path = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/persons1/'

# where these images are to be stored based on their keypoint specifications
persons_img_path_all = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/persons_body/'

# what class to name this file
spec_name='this_type'



# just person samples for 
only_person = False

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

if only_person:
    
    # to get the filtered.json for a specific category simply run 
    # python filter.py --input_json instances_train2017.json --output_json filtered.json --categories person 
    
    dataDir='..'
    dataType='val2017'
    annFile1='annotations/filtered.json'
    coco=COCO(annFile1)

else:
    
    dataDir='..'
    dataType='val2017'
    annFile1='annotations/filtered.json'
    coco1=COCO(annFile1)

    dataDir='..'
    dataType='val2017'
    annFile='annotations/instances_{}.json'.format(dataType)
    coco=COCO(annFile)

annFile = 'annotations/coco_wholebody_train_v1.0.json'
coco_kps=COCO(annFile)
# coco=COCO(annFile)

catIds = coco1.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds_filtered = coco1.getImgIds(catIds=catIds)
filtered_images = coco1.loadImgs(imgIds_filtered)

annotated_dict=coco_kps.anns
image_list=[]
ann_list=[]

for i in annotated_dict.keys():
    temp=annotated_dict[i]
    ann_list.append(annotated_dict[i])
    
    image_list.append(temp['image_id'])

set_image_list=set(image_list)
temp1=list(set_image_list)
image_array=np.array(image_list)

counter = 0
real_index=list([])
label=np.zeros((268015,17,3))
file_names=list([])



for ind , t in enumerate(imgIds_filtered):
    
    if ind < 100000:
        
        im = filtered_images[ind]
        I = cv2.imread(persons_img_path+'coco'+im['file_name'])

        ii = np.where(image_array == t)[0]

        c0=0
        for ind1,k in enumerate(ii):
            print(k)
            print('index: {}, nth person: {}'.format(ind,c0))
            real_index.append(k)
            temp_info=ann_list[k]
            body_points=temp_info['keypoints']
            bod1=np.reshape(body_points,(17,3))

            
            nonzeroind = np.nonzero(body_points) 
            nonzeroind1 = np.nonzero(bod1[0:12,0:2]) 
            save_cond=1
            imres=0
            # if nonzeroind[0].shape[0]>25.5 and I is not None:
            # if c0<10000 and bod1[5,1]*bod1[6,1]*bod1[11,1]*bod1[12,1]>0 and I is not None:
            if bod1[5,1]*bod1[6,1]*bod1[11,1]*bod1[12,1]*bod1[7,1]*bod1[8,1]*bod1[9,1]*bod1[10,1]>0 and I is not None:

                
                # print('index: {}'.format(index))
    
                bbox=np.array(temp_info['bbox']).astype(int)
                enlarge=0.75
                s1=bbox[1]+bbox[3]
                s11=bbox[1]+int(bbox[3]*(1+enlarge))
                s2=bbox[0]+bbox[2]
                s22=bbox[0]+int(bbox[2]*(1+enlarge))
                enlarge=0.5
                
                crop_img = I[max(1,bbox[1]-int(enlarge*bbox[3])):min(s11,I.shape[0]), max(1,int(bbox[0]-enlarge*bbox[2])):min(s22,I.shape[1])]
                

                
                if bbox[3]>25 and bbox[2]>25:
                    
                    bod=np.reshape(body_points,(17,3))
                    visibility=bod1[:,2]
                    bod[:,2]=np.zeros((17,))
                    for x in range(0,17):
                        if bod[x,0]>0 and bod[x,1]>0:
                            bod[x,1]=bod[x,1]-max(1,bbox[1]-int(enlarge*bbox[3]))
                            if bod[x,1]<0:
                                save_cond=0
                                print('Errrorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',x)
                                # time.sleep(2)
                            if bod[x,1]>250:
                                imres=1 
                            bod[x,0]=bod[x,0]-max(1,int(bbox[0]-enlarge*bbox[2]))
                            if bod[x,0]<0:
                                save_cond=0
                                print('Errrorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',x)
                            if bod[x,0]>250:
                                imres=1
                                
                    if save_cond==1:
                        file_names.append(persons_img_path_all+'cropped_'+str(c0).zfill(2)+'_'+str(k).zfill(8)+'_'+im['file_name'])
                        if imres==0:
                            bod[:,2]=visibility

                            # skeleton=bod[:,0:2]
                            # for i in range(17):
                            #     cv2.circle(crop_img, center=tuple(skeleton[i][0:2].astype(int)), radius=1, color=(255, 0, 0), thickness=2)
                            
                            cv2.imwrite(persons_img_path_all+'cropped_'+str(c0).zfill(2)+'_'+str(k).zfill(8)+'_'+im['file_name'],crop_img)
                            fd=1
                        else:

                            scale=max(crop_img.shape[1]/250,crop_img.shape[0]/250)
                            
                            width = int(crop_img.shape[1]/scale)
                            height = int(crop_img.shape[0] /scale)
                            
                            # dsize
                            dsize = (width, height)
                            bod=bod/scale
                            bod[:,2]=visibility

                            for x in range(0,17):
                                if bod[x,0]>250 or bod[x,1]>250:
                                    print('Errrorrrrrrrrrrrrrrrppppppppppppppppprrrrrrrr',x)
                                    # time.sleep(2)
                                    
                            # resize image
                            output = cv2.resize(crop_img, dsize)
                            
                            # skeleton=bod[:,0:2]
                            # for i in range(17):
                            #     cv2.circle(output, center=tuple(skeleton[i][0:2].astype(int)), radius=1, color=(255, 0, 0), thickness=2)
                            
                            cv2.imwrite(persons_img_path_all+'cropped_'+str(c0).zfill(2)+'_'+str(k).zfill(8)+'_'+im['file_name'],output)
              
                                    
                        label[counter,:,:]=bod.astype(float)
                        counter=counter+1
                    
            c0=c0+1
            
            
label1=label[0:counter,:,:]



np.save('data_configs/cocos_'+spec_name+'.npy', label1) 
np.save('data_configs/files_'+spec_name+'.npy',file_names)


