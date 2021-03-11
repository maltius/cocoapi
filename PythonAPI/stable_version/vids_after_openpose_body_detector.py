#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:30:47 2021

@author: mohammad

It reads a video, 

- imports frame by frame, 
- runs openpose,
- obtains midpoints
- aligns the frame
- runs keypoints to obtain them
- saves the final image as the output




"""


import os
os.chdir('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")
    
import argparse
import logging
import os
import numpy as np
import math
import cv2
from scipy import ndimage
import time
import matplotlib.pyplot as plt
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt



os.chdir('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation')


def rot(im_rot,image, xy, a):
    # im_rot = ndimage.rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    # a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center    


def align_im(img,labels):
    
    rot_info={}
    if labels.shape[1]>2.5:
        labels=labels[:,0:2]
    s_max=int(2*max(img.shape))
    if s_max%2==1:
        s_max=s_max+1
    filler=np.zeros((s_max,s_max,3)).astype(np.uint8)
    

    
    # translation
    
    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    stpoint=np.array([int(s_max/2-mid_hip[1]),int(s_max/2-mid_hip[0])])
    filler[stpoint[0]:stpoint[0]+img.shape[0],stpoint[1]:stpoint[1]+img.shape[1],:]=img

    for u in range(labels.shape[0]):
        labels[u,0]=labels[u,0]+stpoint[1]
        labels[u,1]=labels[u,1]+stpoint[0]
    # labels[:,0] += stpoint[1]
    # labels[:,1] += stpoint[0]
    
    rot_info['add10']=-stpoint[1]
    rot_info['add11']=-stpoint[0]

    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[8,0]+labels[11,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    body_vec = mid_hip-mid_sh
    # img = cv2.line(img,tuple(mid_hip),tuple(mid_sh),(255,0,0),5)
    body_vec[1]=-body_vec[1]
    body_vec=-body_vec
    
    angle=np.arcsin(body_vec[0]/(body_vec[0] ** 2+body_vec[1]**2)**0.5)
    angle_deg=math.degrees(angle)
    
    filler_rot = ndimage.rotate(filler, angle_deg,reshape=False,order=0)
    rot_info['rot_im_size']=filler_rot.shape
    rot_info['im_size']=filler.shape
    rot_info['angle']=angle
    
    # if body_vec[0]<0:
    #     angle=angle+90
    mid_hip_old=mid_hip
    for u in range(labels.shape[0]):
        labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
    
    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[8,0]+labels[11,0]),0.5*(labels[8,1]+labels[11,1])]).astype(int)
    
    diam=int(np.linalg.norm(mid_hip-mid_sh))
    final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    


    for u in range(labels.shape[0]):
        # labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
        labels[u,0]=labels[u,0]-(mid_hip[1]-int(diam*1.5))
        labels[u,1]=labels[u,1]-(mid_hip[0]-int(diam*2.2))
        
    rot_info['add20']=(mid_hip[1]-int(diam*1.5))
    rot_info['add21']=(mid_hip[0]-int(diam*2.2))
    
    # labels[:,0] += (-(mid_hip[1]-int(diam*1.5)))
    # labels[:,1] += (-(mid_hip[0]-int(diam*2.2)))


    
    return final,labels,rot_info

# movie_file='/mnt/sda1/downloads/Kavita/Zoom Meeting 2020-12-20 13-15-14.mp4'
movie_file='/mnt/sda1/downloads/Kavita/y2mate.mp4'


w,h=432,368
e = TfPoseEstimator('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/models/graph/cmu/graph_opt.pb', target_size=(w, h))
        

vidcap = cv2.VideoCapture(movie_file)
vid_len=int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

data=np.zeros([vid_len,256,256,3])

success,image = vidcap.read()

for t in range(450):
    success,image = vidcap.read()

c=0
co=0


files = glob.glob('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/main/*.jpg')
for f in files:
    os.remove(f)
    
files = glob.glob('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/resized/*.jpg')
for f in files:
    os.remove(f)


try:    
    os.remove('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/rot_info.npy')
except:
    pass

tot_rot_info={}


while success:
    if co<400:
        orig_image=np.copy(image)
        print('frame: {} from {} frames'.format(str(c),vid_len))
        # frame is read and ready for openpose
        
        
        image = common.conv_imgfile(image, None, None)
        
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4)
        # print(time.time() - tr)
    
    
        # image2 = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)
        temp = TfPoseEstimator.draw_humans1(image, humans, imgcopy=True)
        
        body_len=np.zeros((5))
        for w in range(5):
            try:
                temp_label_dic=temp['human'+str(w+1)]
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
                    
                
        temp_label_dic=temp['human'+str(max_size+1)]
        temp_label=np.zeros((18,2))
        for r in range(18):
            try:
                temp_label[r,0]=temp_label_dic[r][0]
                temp_label[r,1]=temp_label_dic[r][1]
            except:
                pass
            
        temp_label_nonz=temp_label[temp_label[:,0]>0]
        minx=int(0.5*min(temp_label_nonz[:,1]))
        maxx=int(min(1.3*max(temp_label_nonz[:,1]),orig_image.shape[0]))
        miny=int(0.8*min(temp_label_nonz[:,0]))
        maxy=int(min(1.2*max(temp_label_nonz[:,0]),orig_image.shape[1]))
        im_size=max(maxx-minx,maxy-miny)
        img1=image[minx:maxx,miny:maxy,:]
        # plt.imshow(img1)
        
        cv2.imwrite('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/body/sample'+str(co).zfill(5)+'.jpg',img1)
        co += 1
                
        minx=int(min(temp_label[:,1]))
        maxx=int((max(temp_label[:,1])))
        miny=int(min(temp_label[:,0]))
        maxy=int((max(temp_label[:,0])))
        im_size=max(maxx-minx,maxy-miny)
                
        if im_size>300 and temp_label[2,0]*temp_label[5,0]*temp_label[8,0]*temp_label[11,0]>0.5 and False:   
            minx=int(0.5*min(temp_label[:,1]))
            maxx=int(min(1.6*max(temp_label[:,1]),orig_image.shape[0]))
            miny=int(0.8*min(temp_label[:,0]))
            maxy=int(min(1.2*max(temp_label[:,0]),orig_image.shape[1]))
    
            img1=image[minx:maxx,miny:maxy,:]
                
    
            labels2=np.copy(temp_label)
            
            
            labels3=np.copy(temp_label)
            labels3[:,0]=labels3[:,0]-int(0.8*min(labels2[:,0]))
            labels3[:,1]=labels3[:,1]-int(0.5*min(labels2[:,1]))
            
            # aa=time.time()
            img4,labels4,rot_info_1=align_im(img1, np.copy(labels3))
            
            rot_info_1['final_size']=img4.shape
            rot_info_1['add30']=int(0.8*min(labels2[:,0]))
            rot_info_1['add31']=int(0.5*min(labels2[:,1]))       
            
            if min(img4.shape[0:2])>100 and cv2.countNonZero(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY))>img4.shape[0]*img4.shape[1]/10:
                
                # it is ready for blazepose as img4  and grod truth of labels are    new_labels[c,:,0:2]=labels4.astype(float)
                img_5=(cv2.resize(img4,(256,256)))
    
                cv2.imwrite('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/resized/sample'+str(co).zfill(5)+'.jpg',img_5)
                cv2.imwrite('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/main/sample'+str(co).zfill(5)+'.jpg',orig_image)
    
                tot_rot_info['sample'+str(co).zfill(5)]=rot_info_1
                
                co += 1
    
        
        else:
            print('no human')
        
        success,image = vidcap.read()
    
        c += 1
    else:
        break

np.save('/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/rot_info.npy',tot_rot_info)





