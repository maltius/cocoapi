import os
import tensorflow as tf
from pynvml import *
import tensorflow.keras as keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import h5py


path='pre_processed_HM/files_mids_new_aligned_sc2_pc_no0ing/'

keyp=17 
X_train_filenames1=np.load('data_configs/files_17.npy')
label11= np.load('data_configs/cocos_17.npy')

dev_set=X_train_filenames1[7501:7501+600]

X_train_filenames1=np.load('data_configs/files_mids_new_aligned_sc2_pc.npy')  
label2= np.load('data_configs/cocos_mids_new_aligned_sc2_pc_no0ing.npy')
# label2= np.load('data_configs/cocos_17_new_aligned.npy')

label2=label2[0:X_train_filenames1.shape[0],:,:]

to_be_omitted=list()
for t in range(X_train_filenames1.shape[0]-1,0,-1):
    str1=X_train_filenames1[t]
    for r in range(dev_set.shape[0]-1):
        str2=dev_set[r]
        if str1[len(str1)-36:]==str2[len(str2)-36:]:
            to_be_omitted.append(t)
            
X_train_filenames2=np.delete(X_train_filenames1,to_be_omitted)

label3=np.zeros((X_train_filenames2.shape[0],keyp,3))
c=0
for t in range(label2.shape[0]): 
    if t not in to_be_omitted:
        label3[c,:,:]=label2[t,:,:]
        c=c+1
print(c-X_train_filenames2.shape[0])

in_sam=10010
I=cv2.imread(X_train_filenames2[in_sam]).astype(np.uint8)
plt.axis('off')
plt.imshow(I)
plt.show()

skeleton = label3[in_sam,:,:].astype(int)
for i in range(keyp):
    cv2.circle(I, center=tuple(skeleton[i][0:2]), radius=1, color=(255, 0, 0), thickness=2)
plt.axis('off')
plt.imshow(I)
plt.show()

time.sleep(5)

filenames=X_train_filenames2
labels=label3

train_mode=0

def getGaussianMap(joint = (16, 16), heat_size = 128, sigma = 2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size),dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    g.shape
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    """
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    cv2.imshow("debug", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return heatmap

    
if True:
    for idx in range(int(filenames.shape[0]/128)):
        batch_x = filenames[idx * 128 : (idx+1) * 128]
        batch_y = labels[idx * 128 : (idx+1) * 128,:,:]
        # print(np.array(batch_x).shape)
        # print(tf.shape(batch_y))
        
        
        label=np.copy(batch_y)
        # tot_data=int(tf.shape(batch_y)[0])
        tot_data=batch_y.shape[0]
        # print(tot_data)
        # time.sleep(5)
        
        # data_bat=np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #        for file_name in batch_x])/255.0
        
        data = np.zeros([tot_data, 256, 256, 3])
        if train_mode==0:
            heatmap_set = np.zeros((tot_data, 128, 128, 17), dtype=np.float32)
        
        to_be_omitted=list()
        reduce_it=0
        for i in range(tot_data):
            try:
                if i==0:
                    print('processing batch')
                # FileName = "./dataset/lsp/images/im%04d.jpg" % (i + 1)
                FileName = batch_x[i]
                # FileName=FileName[0:45]+"aligned_"+FileName[45:]
                # ii=cv2.imread(FileName)
                img = tf.io.read_file(FileName)
                img = tf.image.decode_image(img)
                img_shape = img.shape
                # Attention here img_shape[0] is height and [1] is width
        
                if min(img_shape[0:2])<30:
                    print(num_of_joints)
                label[i, :, 0] *= (256 / img_shape[1])
                label[i, :, 1] *= (256 / img_shape[0])
                # generate heatmap set
                # for j in range(17):
                #     _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
                #     # print(_joint)
                #     heatmap_set[i, :, :, j] = getGaussianMap(joint = _joint, heat_size = 128, sigma = 4)
                    
                if train_mode==0:
                    for j in range(17):
                        _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
                        # print(_joint)
                        heatmap_set[i, :, :, j] = getGaussianMap(joint = _joint, heat_size = 128, sigma = 4)
                data[i] = tf.image.resize(img, [256, 256])
            
            except:
                pass
                to_be_omitted.append(i)
                reduce_it=1
                print(i)
                
        if reduce_it==1:
            data1= np.zeros([tot_data-len(to_be_omitted), 256, 256, 3])
            if train_mode==0:
                heatmap_set1 = np.zeros((tot_data-len(to_be_omitted), 128, 128, 17), dtype=np.float32)
                label1 = np.zeros((label.shape[0]-len(to_be_omitted),label.shape[1],label.shape[2]))

            else:
                label1 = np.zeros((label.shape[0]-len(to_be_omitted),label.shape[1],label.shape[2]))
            
            c=0
            for w in range(data.shape[0]):
                if w not in to_be_omitted:
                    data1[c,:,:,:]=data[w,:,:,:]
                    if train_mode==0:
                        heatmap_set1[c,:,:,:]=heatmap_set[w,:,:,:]
                        label1[c,:,:]=label[w,:,:]

                    else:
                        label1[c,:,:]=label[w,:,:]
                    c=c+1

            
        # print()            
        print(label.shape)
        # print(label1.shape)
        
        if reduce_it==0:
            np.save(path+'heat_'+str(idx).zfill(6)+'.npy',heatmap_set)
            np.save(path+'data_'+str(idx).zfill(6)+'.npy',data)

            np.save(path+'label_'+str(idx).zfill(6)+'.npy',label)
            
            # with h5py.File(path+'heat_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=heatmap_set)
                
            # with h5py.File(path+'data_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=data)
                
            # with h5py.File(path+'label_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=label)
        else:
            # with h5py.File(path+'heat_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=heatmap_set1)
                
            # with h5py.File(path+'data_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=data1)
                
            # with h5py.File(path+'label_'+str(idx).zfill(6)+'.h5', 'w') as hf:
            #     hf.create_dataset("name-of-dataset",  data=label1)
            np.save(path+'heat_'+str(idx).zfill(6)+'.npy',heatmap_set1)
            np.save(path+'data_'+str(idx).zfill(6)+'.npy',data1)

            np.save(path+'label_'+str(idx).zfill(6)+'.npy',label1)
            