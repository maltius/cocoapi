import argparse
import logging
import sys
import time
import os
import json
import imagesize
import matplotlib.pylab as plt
import matplotlib as mpl
import math


import cv2
import numpy as np


paths='/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'


path_im='/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
path1='/mnt/sda1/downloads/action/'
jfiles = [f for f in os.listdir(path_im) if f.endswith('.jpg')]
filesnames=list([])
    
path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/'
jpeg_path = '/mnt/sda1/downloads/ai-challenger/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
file_name = 'keypoint_train_annotations_20170902.json'
    # path_save = '/mnt/sda1/downloads/cocoapi-master/PythonAPI/aic_persons_17_single/'

labels_raw = json.loads(open(os.path.join(path,file_name)).read())
   
height=list()
width=list()
ratio=list()
im_height=list()
im_width=list()
norm_hight=list()
norm_width=list()
cen_h=list()
cen_w=list()
boxes_list=list()


for t in range(len(labels_raw)):
    
    for j in range(len(labels_raw[t]['human_annotations'].keys())):
        try:
            # I=cv2.imread(path_im+labels_raw[t]['image_id']+'.jpg')
            
            im_width1, im_height1 = imagesize.get(path_im+labels_raw[t]['image_id']+'.jpg')
            
            # width, height = imagesize.get("test.png")
            label=labels_raw[t]['keypoint_annotations']['human'+str(j+1)]
            labels=np.array(label).reshape(14,3)
            labels=labels[:,0:2]
            
            if t%50000==0:
                print(str(t)+'  number of files is processed out of '+str(len(jfiles)))
            if labels[12,0]*labels[13,0]*labels[6,0]*labels[9,0]>0 and np.linalg.norm(labels[6,:]-labels[9,:])>5:
                box_ind=[6,9,12]
                label_box=labels[box_ind,:]
                
                minx=int(min(label_box[:,1]))
                maxx=int((max(label_box[:,1])))
                miny=int(min(label_box[:,0]))
                maxy=int((max(label_box[:,0])))
                box_size=[maxx-minx,maxy-miny]
                
                # mid_down=(labels[6,:]+labels[9,:])/2
                height.append(box_size[0])
                width.append(box_size[1])
                ratio.append(box_size[1]/box_size[0])
                im_height.append(im_height1)
                im_width.append(im_width1)
                norm_hight.append(box_size[0]/im_height1)
                norm_width.append(box_size[1]/im_width1)
                cen_h.append(np.mean((maxx,minx))/im_height1)
                cen_w.append(np.mean((maxy,miny))/im_width1)
                boxes_list.append((minx/im_height1,maxx/im_height1,miny/im_width1,maxy/im_width1))
                
                
                
                
        except:
            pass
                
        
from sklearn.cluster import KMeans


X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])

X=np.zeros((len(norm_width),2))

for i in range(len(norm_width)):
    X[i,0]=norm_hight[i]
    X[i,1]=norm_width[i]

kmeans = KMeans(n_clusters=112, random_state=0).fit(X)

kmeans.labels_array([1, 1, 1, 0, 0, 0], dtype=int32)
kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
kmeans.cluster_centers_
array([[10.,  2.],
       [ 1.,  2.]])

mpl.rcParams['agg.path.chunksize'] = 10000

plt.scatter(norm_hight,norm_width)

# rng = np.random.RandomState(10)  # deterministic random data

# a = np.hstack((rng.normal(size=1000),

#                rng.normal(loc=5, scale=2, size=1000)))

# _ = plt.hist(np.array(ratio), bins=np.arange(12) ) # arguments are passed to np.histogram

# _ = plt.hist(np.array(ratio), bins=np.arange(0,1,0.1) ) # arguments are passed to np.histogram

# # _ = plt.hist(np.array(ratio), bins='auto' ) # arguments are passed to np.histogram

# plt.title("Histogram with 'auto' bins")

# plt.show()



# hist, xedges, yedges = np.histogram2d(height, width, bins=10, range=[[0, 1000], [0, 1000]])

# # Construct arrays for the anchor positions of the 16 bars.
# xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = 0

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Construct arrays with the dimensions for the 16 bars.
# dx = dy = 0.5 * np.ones_like(zpos)
# dz = hist.ravel()

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# plt.show()


# create a set of boxes:
if False:

    boxes10=list()
    
    for i in range(len(norm_width)):
        if i%50000==0:
            print(i)
        base=10
        divi=10/8
        adad=1000*math.floor(boxes_list[i][0]*base/divi)+100*math.ceil(boxes_list[i][1]*base/divi)+10*math.floor(boxes_list[i][2]*base/divi)+1*math.ceil(boxes_list[i][3]*base/divi)
        
        boxes10.append(adad)
from collections import Counter
a = dict(Counter(boxes10))
my_tup=sorted(a.items(), key=lambda x:x[1])
        

        
final_896=my_tup[-896:]
        
prior_boxes = np.zeros((896,4))  
for i in range(895,-1,-1):
    # pass
    multip=divi/10
    test_no=final_896[i][0]
    x1=math.floor(test_no/1000)*multip
    if x1<=0: 
        x1=0.01
    test_no -= math.floor(test_no/1000)*1000
    x2=math.floor(test_no/100)*multip+0.01
    if x2>1:
        x2=1
    test_no -= math.floor(test_no/100)*100
    y1=math.floor(test_no/10)*multip-0.01
    if y1<=0: 
        y1=0.01
    test_no -= math.floor(test_no/10)*10
    y2=math.floor(test_no/1)*multip+0.01
    if y2>1:
        y2=1
    prior_boxes[i,:]=[x1 ,y1 ,x2 ,y2]
    
    