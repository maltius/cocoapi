import os
import numpy as np
import math
import cv2
from scipy import ndimage
import time
import matplotlib.pyplot as plt

label1= np.load('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/action_pre_aligned_humans_coco.npy',allow_pickle=True)
file_name = np.load('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/files_action_pre_aligned_humans_coco.npy')
p_a='/mnt/sda1/downloads/cocoapi-master/PythonAPI/aligned_action_persons_mids/'

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
    
    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[11,0]+labels[8,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
    stpoint=np.array([int(s_max/2-mid_hip[1]),int(s_max/2-mid_hip[0])])
    filler[stpoint[0]:stpoint[0]+img.shape[0],stpoint[1]:stpoint[1]+img.shape[1],:]=img

    for u in range(labels.shape[0]):
        labels[u,0]=labels[u,0]+stpoint[1]
        labels[u,1]=labels[u,1]+stpoint[0]
    # labels[:,0] += stpoint[1]
    # labels[:,1] += stpoint[0]
    
    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[8,0]+labels[11,0]),0.5*(labels[11,1]+labels[8,1])]).astype(int)
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
    
    mid_sh=np.array([0.5*(labels[2,0]+labels[5,0]),0.5*(labels[2,1]+labels[5,1])]).astype(int)
    mid_hip=np.array([0.5*(labels[8,0]+labels[11,0]),0.5*(labels[8,1]+labels[11,1])]).astype(int)
    
    diam=int(np.linalg.norm(mid_hip-mid_sh))
    final=filler_rot[mid_hip[0]-int(diam*2.2):mid_hip[0]+int(diam*2.2),mid_hip[1]-int(diam*1.5):mid_hip[1]+int(diam*1.7),:]
    


    for u in range(labels.shape[0]):
        # labels[u,:]=rot(filler_rot,filler,labels[u,:],angle)
        labels[u,0]=labels[u,0]-(mid_hip[1]-int(diam*1.5))
        labels[u,1]=labels[u,1]-(mid_hip[0]-int(diam*2.2))

    # labels[:,0] += (-(mid_hip[1]-int(diam*1.5)))
    # labels[:,1] += (-(mid_hip[0]-int(diam*2.2)))


    
    return final,labels


# label1= np.load('data_configs/cocos_mids_new_aligned_pc.npy')
# file_name = np.load('data_configs/files_mids_new_aligned_pc.npy')




new_file_name=list()
label=np.zeros((file_name.shape[0],18,2))
new_label=np.copy(label)

tot_data=label.shape[0]


aa=time.time()
bb=time.time()

omitted_list=list()
new_labels=np.zeros((file_name.shape[0],18,2))

c=0
for i in range(tot_data):
    try:
    # if True:
        if c<10000000:
            if i%100==0:
                print(i)
                print('just for that: {}'.format((time.time()-aa)))
                print('just for that: {}'.format((time.time()-bb)))
    
                aa=time.time()
            # FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
            FileName = file_name[i]
            # ii=cv2.imread(file_name[i])
            
            
            # if img.shape
            body_len=np.zeros((5))
            for w in range(5):
                try:
                    temp_label_dic=label1[i]['human'+str(w+1)]
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
                
            
            temp_label_dic=label1[i]['human'+str(max_size+1)]
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
            
            if im_size>300 and temp_label[2,0]*temp_label[5,0]*temp_label[8,0]*temp_label[11,0]>0.5:   
                img = cv2.imread(FileName)
    
                        
    
    
                minx=int(0.5*min(temp_label[:,1]))
                maxx=int(min(1.6*max(temp_label[:,1]),img.shape[0]))
                miny=int(0.8*min(temp_label[:,0]))
                maxy=int(min(1.2*max(temp_label[:,0]),img.shape[1]))
                # im_size=max()
    
                img1=img[minx:maxx,miny:maxy,:]
                    
    
                labels2=np.copy(temp_label)
                
                # skeleton=labels2
                # for ii in range(18):
                #     cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=5, color=(255, 0, 0), thickness=5)
                #     cv2.putText(img,str(ii), tuple(skeleton[ii][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),thickness=3)
                # plt.imshow(img)
                # plt.imshow(img4)
                
                # skeleton=labels4
                # for ii in range(18):
                #     cv2.circle(img4, center=tuple(skeleton[ii][0:2].astype(int)), radius=10, color=(0, 255, 0), thickness=20)
                #     cv2.putText(img4,str(ii), tuple(skeleton[ii][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255))
                
                labels3=np.copy(temp_label)
                labels3[:,0]=labels3[:,0]-int(0.8*min(labels2[:,0]))
                labels3[:,1]=labels3[:,1]-int(0.5*min(labels2[:,1]))
                
                # aa=time.time()
                img4,labels4=align_im(img1, np.copy(labels3))
                # time.time()-aa
                # FileNames=FileName[0:45]+"aligned_"+FileName[45:]
                # FileNames=FileName[0:45]+"aligned_"+FileName[45:]
                if min(img4.shape[0:2])>100 and cv2.countNonZero(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY))>img4.shape[0]*img4.shape[1]/10:
                    FileNames=p_a+FileName[34:]
                    
                    cv2.imwrite(FileNames, img4)
                    # for k in range(labels2.shape[0]):
                    #     if label[i,k,2]==0: 
                    #         labels2[k,0]=0
                    #         labels2[k,1]=0
        
                    new_labels[c,:,0:2]=labels4.astype(float)
                    # new_labels[c,:,2]=label[i,:,2].astype(float)
                    new_file_name.append(FileNames)
            
                    c=c+1
                else:
                    omitted_list.append(i)

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

np.save('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/action_aligned_humans_coco.npy',new_labels1)
np.save('/datadrive/downloads/cocoapi-master/PythonAPI/openpose/tf-pose-estimation/data_configs_op/files_action_aligned_humans_coco.npy',np.array(new_file_name))  

# np.save('data_configs/mpii_croped.npy',new_labels1)
# np.save('data_configs/files_mpii_cropped.npy',np.array(new_file_name))  

i=25800
FileName = new_file_name[i]

img = cv2.imread(FileName)
plt.imshow(img)    

skeleton=new_labels1[i,:,:]
for ii in range(18):
    # cv2.circle(img, center=tuple(skeleton[ii][0:2].astype(int)), radius=2, color=(0, 255, 0), thickness=20)
    cv2.putText(img,str(ii), tuple(skeleton[ii][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),thickness=5)
        
plt.imshow(img)    



