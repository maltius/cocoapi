import os
import platform
import numpy as np
import math
import cv2
from scipy import ndimage
import time

# read files and labels
label1= np.load('data_configs/cocos_mids_new_aligned_pc.npy')
file_name = np.load('data_configs/files_mids_new_aligned_pc.npy')

# what to name the file
spec_name='what_to_call_it'

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

    # labels[:,0] += (-(mid_hip[1]-int(diam*1.5)))
    # labels[:,1] += (-(mid_hip[0]-int(diam*2.2)))


    
    return final,labels




# label1= np.load('data_configs/mpii_raw.npy')
# file_name = np.load('data_configs/files_raw.npy')

new_file_name=list()
label=label1[0:file_name.shape[0],0:17,:]
new_label=np.copy(label)

# read images
tot_data=label.shape[0]


aa=time.time()
bb=time.time()

omitted_list=list()
new_labels=np.zeros((len(file_name),label1.shape[1],3))

c=0
for i in range(tot_data):
    if c<1000000:
        try:
            
            if i%100==0:
                print(i)
                print('just for that: {}'.format((time.time()-aa)))
                print('just for that: {}'.format((time.time()-bb)))
    
                aa=time.time()
            # FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
            FileName = file_name[i]
            # ii=cv2.imread(file_name[i])
            img = cv2.imread(FileName)
            labels=np.copy(label[i,:,:])
            img1,labels2=align_im(img, np.copy(label[i,:,:]))
            FileNames=FileName[0:45]+"aligned_"+FileName[45:]
            # FileNames=FileName[0:33]+"aligned_"+FileName[33:]
    

            new_labels[c,:,0:2]=labels2.astype(float)
            new_labels[c,:,2]=label[i,:,2].astype(float)
            new_file_name.append(FileNames)
    
            c=c+1
            # new_label[i,:,2]=np.zeros((new_label.shape[1],))          
     
        except:
            print('none')
            omitted_list.append(i)
        
new_labels1=new_labels[0:c]

# new_labels=np.zeros((len(new_file_name),new_label.shape[1],3))

# c=0
# for t in range(len(file_name)): 
#     if t not in omitted_list:
#         new_labels[c,:,:]=new_label[t,:,:]
#         c=c+1
# print(c-len(new_file_name))

        
np.save('data_configs/cocos_aligned_'+spec_name+'.npy',new_labels)
np.save('data_configs/files_aligned'+spec_name+'.npy',np.array(new_file_name))  

