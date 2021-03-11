"""
Created on Tue Nov 17 20:04:17 2020

@author: altius
"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance

from config import total_epoch, train_mode

from model import BlazePose

import os
import platform
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import math
from config import num_joints, batch_size, gaussian_sigma, gpu_dynamic_memory
import cv2
from scipy import ndimage
import time
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

def align_im(img,labels):
    
    if labels.shape[1]>2.5:
        labels=labels[:,0:2]
    s_max=int(2*max(img.shape))
    if s_max%2==1:
        s_max=s_max+1
    filler=np.zeros((s_max,s_max,3)).astype(np.uint8)
    

    
    # translation
    
    mid_hip=np.array([0.5*(labels[11,0]+labels[12,0]),0.5*(labels[11,1]+labels[12,1])]).astype(int)
    mid_sh=np.array([0.5*(labels[5,0]+labels[6,0]),0.5*(labels[5,1]+labels[6,1])]).astype(int)
    stpoint=np.array([int(s_max/2-mid_hip[1]),int(s_max/2-mid_hip[0])])
    filler[stpoint[0]:stpoint[0]+img.shape[0],stpoint[1]:stpoint[1]+img.shape[1],:]=img

    for u in range(labels.shape[0]):
        labels[u,0]=labels[u,0]+stpoint[1]
        labels[u,1]=labels[u,1]+stpoint[0]
    # labels[:,0] += stpoint[1]
    # labels[:,1] += stpoint[0]
    
    mid_hip=np.array([0.5*(labels[11,0]+labels[12,0]),0.5*(labels[11,1]+labels[12,1])]).astype(int)
    mid_sh=np.array([0.5*(labels[5,0]+labels[6,0]),0.5*(labels[5,1]+labels[6,1])]).astype(int)
    body_vec = mid_hip-mid_sh
    img = cv2.line(img,tuple(mid_hip),tuple(mid_sh),(255,0,0),5)
    body_vec[1]=-body_vec[1]
    body_vec=-body_vec
    
    angle=np.arcsin(body_vec[0]/(body_vec[0] ** 2+body_vec[1]**2)**0.5)
    angle_deg=math.degrees(angle)
    
    filler_rot = ndimage.rotate(filler, angle_deg,reshape=False,order=0)
    
    # if body_vec[0]<0:
    #     angle=angle+90
    mid_hip_old=mid_hip
    for u in range(labels.shape[0]):
        labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
    
    mid_hip=np.array([0.5*(labels[11,0]+labels[12,0]),0.5*(labels[11,1]+labels[12,1])]).astype(int)
    mid_sh=np.array([0.5*(labels[5,0]+labels[6,0]),0.5*(labels[5,1]+labels[6,1])]).astype(int)
    
    diam=int(np.linalg.norm(mid_hip-mid_sh))
    final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    


    for u in range(labels.shape[0]):
        # labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
        labels[u,0]=labels[u,0]-(mid_hip[1]-int(diam*1.5))
        labels[u,1]=labels[u,1]-(mid_hip[0]-int(diam*2.2))
        
    return final,labels



inde=5
path='C:/Users/altius/Documents/backs/MATLAB/June_23/Zleft/'
jfiles = [f for f in os.listdir(path) if f.endswith('.json')]

path2='C:/Users/altius/Documents/backs/MATLAB/Data_Gallery_room_22_06/Zleft/'
path3='C:/Users/altius/Documents/backs/MATLAB/Data_Gallery_room_22_06/aligned_Zleft/'


jpegfiles = [f for f in os.listdir(path2) if f.endswith('.jpg')]

I = cv2.imread(os.path.join(path2,jpegfiles[0]))

train_data = np.zeros([len(jpegfiles), 256, 256, 3])

for i in range(0,len(jfiles),10):
    print(i)
    I = cv2.imread(os.path.join(path2,jpegfiles[i]))

    json_data1 = json.loads(open(os.path.join(path,jfiles[i])).read()) 
    locs=json_data1['people'][0]['pose_keypoints_2d']
    
    
    # for j in range(int(len(locs)/3)):
    #     if True:
    #         # J=np.copy(I)
    #         locs1=list((int(locs[3*j]),int(locs[3*j+1])))
    #         cv2.circle(I, center=tuple(locs1), radius=3, color=(255, 0, 0), thickness=5);
    #         # plt.title(str(j))

    # plt.imshow(I)
    # plt.show()
    # time.sleep(2)
    
    
    
    lebel=np.ones((17,3))
    lebel[5,0]=locs[6]
    lebel[5,1]=locs[7]
    
    lebel[6,0]=locs[15]
    lebel[6,1]=locs[16]
    
    lebel[11,0]=locs[27]
    lebel[11,1]=locs[28]
    
    lebel[12,0]=locs[36]
    lebel[12,1]=locs[37]

    img1,labels2=align_im(I, np.copy(lebel[:,:]))
    
    cv2.imwrite(os.path.join(path3,jpegfiles[i]),img1)
    
path3='C:/Users/altius/Documents/backs/MATLAB/Data_Gallery_room_22_06/aligned_Zleft/'
jpegfiles = [f for f in os.listdir(path3) if f.endswith('.jpg')]

train_data = np.zeros([len(jpegfiles), 256, 256, 3])


for i in range(0,len(jpegfiles)):
    
    img = tf.io.read_file(os.path.join(path3,jpegfiles[i]))
    img = tf.image.decode_image(img)

    train_data[i] = tf.image.resize(img, [256, 256])

model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

checkpoint_path = "training_checkpoints_new_aligned_batches_precalc/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=108))


y = model.predict(train_data)

ex_labels=np.zeros((y.shape[0],y.shape[3],3))
for p in range(y.shape[0]):   
    for q in range(y.shape[3]):
        try_var=(y[p,:,:,q])
        # res=try_var.argmax(axis=(1))
        ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
        ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2

path4='C:/Users/altius/Documents/backs/MATLAB/Data_Gallery_room_22_06/res_Zleft'

for p in range(0,y.shape[0]):
    skeleton = ex_labels[p,:,0:2].astype(int)
    img = train_data[p].astype(np.uint8)
    plt.imshow(img)
    for i in range(17):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
    
    cv2.imwrite(os.path.join(path4,jpegfiles[p]),img)
    
    

    for j in range(int(len(locs)/3)):
        if True:
            J=np.copy(I)
            locs1=list((int(locs[3*j]),int(locs[3*j+1])))
            cv2.circle(J, center=tuple(locs1), radius=15, color=(255, 0, 0), thickness=5);
            plt.title(str(j))
            plt.imshow(J)
            plt.show()
            time.sleep(3)
            



import cv2
import numpy as np
import os
npathIn= path4
pathOut = os.path.join(path4,'video.mp4')
fps = 5

jpegfiles1 = [f for f in os.listdir(path4) if f.endswith('.jpg')]


frame_array=list()
for i in range(len(jpegfiles1)):
    filename=os.path.join(path4, jpegfiles1[i])    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
    
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
# writing to a image array
    out.write(frame_array[i])
out.release()



