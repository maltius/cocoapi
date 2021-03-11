#!~/miniconda3/envs/tf2/bin/python
import os
import tensorflow as tf
from config import total_epoch, train_mode

from model import BlazePose
import timeit
import numpy as np
import cv2
import time
import math
from scipy import ndimage
import matplotlib.pyplot as plt
# from data_coco_17_aligned import train_dataset, test_dataset, data, label, heatmap_set

from data_aic_10_rot_val import train_heatmap_set, test_heatmap_set, train_label, test_label, train_data, test_data


keyp=10
output_samples='any'
sample_save=False

# sha=np.load(checkpoint_dir+'/history.npy',allow_pickle=True).item()

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def rot(im_rot,image, xy, a):
    # im_rot = ndimage.rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    # a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return new+rot_center    

def coco_pck_amp(est_keys,true_keypoints):
    dist=1000
    torso_diam=np.linalg.norm(true_keypoints[8,0:2] - true_keypoints[9,0:2])
    est_key=est_keys[:,0:2]
    true_keypoint=true_keypoints[:,0:2]
    
    dist_all= np.array([ np.linalg.norm(true_keypoint[x,:] - est_key[x,:]) for x in range(est_key.shape[0])])
    
    
    return np.sum(dist_all)

def bone(skeleton,img,keyp):
    if skeleton.shape[1]==3:
        skeleton=skeleton[:,0:2]
    bonepairs=[(0,6),(0,5),(5,7),(7,9),(6,8),(8,10),(11,12),(11,13),(13,15),(12,14),(14,16)]
    bone_pairs1=list()
 
    for bones in bonepairs:
        cv2.line(img, (skeleton[bones[1],0], skeleton[bones[1],1]), (skeleton[bones[0],0], skeleton[bones[0],1]), (255, 0, 0), thickness=2)
    return img



model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()])

# checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_aic_LR_001_single_rot90/ckpt_{epoch}"
# checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_aic_LR_001_single_rot/ckpt_{epoch}"
# model.load_weights(checkpoint_path.format(epoch=28))


checkpoint_path = "training_checkpoints_new_aligned_batches_precalc_no0ing_aic_LR_001_single_rot_11/ckpt_{epoch}"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path.format(epoch=20))





# model.evaluate(test_dataset)


if sample_save==True:    
 
    
    if train_mode:
        
        true_keys= label #np.load('data_configs/cocos_17.npy')
        
        # training data
        
        y = np.zeros((100, keyp, 3)).astype(np.uint8)
        y[0:100] = model(train_data[0:100]).numpy().astype(np.uint8)
        # y[1000:2000] = model(data[1000:2000]).numpy().astype(np.uint8)
        for t in range(100):
            tt=0
            skeleton = y[t]
            print(skeleton)
            img = data[t+tt].astype(np.uint8)
            for i in range(keyp):
                cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            # bone(skeleton,img,keyp)
            cv2.imwrite("./results_train_aligned/face_rot_%d.jpg"%t, img)
            cv2.imshow("test", img)
            plt.imshow(img)
            # plt.savefig("demo_"+str(t)+".png")
            cv2.waitKey(50)
            pass
        cv2.destroyAllWindows()
        
        # testing data
        
        y = np.zeros((100,keyp, 3)).astype(np.uint8)
        test_sam1=data[data.shape[0]-101:-1]
        # test_sam_rot=test_sam[-1,:,:,:]
        # img_rotate_90_counterclockwise = cv2.rotate(test_sam_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # test_sam[-1,:,:,:]=img_rotate_90_counterclockwise
        y[0:100] = model(test_sam1).numpy().astype(np.uint8)
        
        # y[1000:2000] = model(data[1000:2000]).numpy().astype(np.uint8)
        for t in range(100):
            tt=0
            skeleton = y[t]
            print(skeleton)
            img = data[data.shape[0]-101+t].astype(np.uint8)
            for i in range(keyp):
                cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            bone(skeleton,img,keyp)
            cv2.imwrite("./result_test_aligned/face_rot_%d.jpg"%t, img)
            cv2.imshow("test", img)
            # plt.savefig("demo_"+str(t)+".png")
            cv2.waitKey(50)
            pass
        cv2.destroyAllWindows()
        
    else:
        
        
        st=timeit.default_timer()
        # y = model.predict(test_dataset)
        y = model.predict(data[1900:2000])
        print(timeit.default_timer()-st)
    
        import matplotlib.pyplot as plt
        import numpy as np    
        for t in range(10):
            plt.figure(figsize=(8,8), dpi=150)
            for i in range(14):
                plt.subplot(4, 4, i+1)
                # cv2.imshow()
                # plt.imshow(np.uint8(data[1999]))
                # plt.hold(True)
    
                plt.imshow(y[:, :, i].astype(np.uint8))
            plt.savefig("demo"+str(t)+".png")
            plt.show()
    pass
    
   
    
# true_keys= label

# test_end_ind=data.shape[0]
# test_st_ind=7501
# test_ind=np.array(range(test_st_ind,test_end_ind))
# test_set_size=test_ind.shape[0]


for chk in range(0,1):
    if True:

        try:
            
            # y = np.zeros((test_set_size,128,128, keyp)).astype(np.uint8)
            test_sam1=test_data[2000:]
            t_label=test_label[2000:]
            # y = model.predict(data[1990:2000])
                # y = model.predict(data[1990:2000])


            st=time.time()
            y = model.predict(test_sam1)
            ed=time.time()
            pck_blazepose_test=np.zeros((1,y.shape[0]))

            # print('Processed each image in {} mili-seconds using GPU'.format(int(1000*(ed-st)/test_set_size)))
            
            ex_labels=np.zeros((y.shape[0],y.shape[3],3))
            for p in range(y.shape[0]):   
                for q in range(y.shape[3]):
                    try_var=(y[p,:,:,q])
                    # res=try_var.argmax(axis=(1))
                    ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
                    ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
            
            
            # for p in range(0,y.shape[0],10):
            #     skeleton = ex_labels[p,:,0:2].astype(int)
            #     img = test_sam1[p].astype(np.uint8)
            #     for i in range(keyp):
            #         cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            #     skeleton=t_label[p].astype(int)

            #     for i in range(keyp):
            #         cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 0, 255), thickness=2)
            #     # plt.imshow(img)
            #     cv2.imwrite("./results_test_aligned_80k_no0ing/coco_%d.jpg"%p, img)

                
                
            
            for t in range(y.shape[0]):
                
                pck_blazepose_test[chk,t]=np.mean(coco_pck_amp(ex_labels[t,:,:],t_label[t]))
                
                if  pck_blazepose_test[chk,t]<8:
                   im=test_sam1[t].astype(np.uint8)
                   
                   skeleton = t_label[t,:,0:2].astype(int)
                   for ii in range(keyp):
                       cv2.circle(im, center=tuple(skeleton[ii][0:2]), radius=2, color=(255, 0, 0), thickness=2)
                   
                   skeleton = ex_labels[t,:,:].astype(int)
                   
                   for ii in range(keyp):
                       cv2.circle(im, center=tuple(skeleton[ii][0:2]), radius=2, color=(0, 255, 0), thickness=2)
                       
                   cv2.imwrite('errorh/img_'+str(t)+'.jpeg',im)

                    
                else:
                    im=test_sam1[t].astype(np.uint8)
            
                    skeleton = t_label[t,:,:].astype(int)
                    for ii in range(keyp):
                        cv2.circle(im, center=tuple(skeleton[ii][0:2]), radius=2, color=(255, 0, 0), thickness=2)
                    
                    skeleton = ex_labels[t,:,:].astype(int)
                    for ii in range(keyp):
                        cv2.circle(im, center=tuple(skeleton[ii][0:2]), radius=2, color=(0, 255, 0), thickness=2)
                            
                    cv2.imwrite('errorh1/img_'+str(t)+'.jpeg',im)


            pck_blazepose_test_avg=np.mean(pck_blazepose_test)

            
            if False:
                test_sam1=train_data[0:1000]
                t_label=train_label[0:1000]
                # y = model.predict(data[1990:2000])
                    # y = model.predict(data[1990:2000])
                pck_blazepose_train=np.zeros((1,y.shape[0]))
    
    
                st=time.time()
                y = model.predict(test_sam1)
                ed=time.time()
            
            # print('Processed each image in {} mili-seconds using GPU'.format(int(1000*(ed-st)/test_set_size)))
            
                ex_labels=np.zeros((y.shape[0],y.shape[3],3))
                for p in range(y.shape[0]):   
                    for q in range(y.shape[3]):
                        try_var=(y[p,:,:,q])
                        # res=try_var.argmax(axis=(1))
                        ind = np.unravel_index(np.argmax(try_var, axis=None), try_var.shape)  # returns a tuple
                        ex_labels[p,q,0:2]=np.array(list([ind[1],ind[0]]))*2
                
                for t in range(y.shape[0]):
                    
                    pck_blazepose_train[chk,t]=np.mean(coco_pck_amp(ex_labels[t,:,:],t_label[t]))
                pck_blazepose_train_avg=np.mean(pck_blazepose_train)

            if False:
                # sth=500
                y = np.zeros((sth-1,keyp, 3)).astype(np.uint8)
                test_sam1=data[0:sth-1:]
                
                st=time.time()
                y[0:y.shape[0]] = model(test_sam1).numpy().astype(np.uint8)
                ed=time.time()
                pck_blazepose_train=np.zeros((1,y.shape[0]))

                
                print('Processed each image in {} mili-seconds using GPU'.format(int(1000*(ed-st)/test_set_size)))
                
                
                for t in range(y.shape[0]):
                
                    pck_blazepose_train[chk,t]=np.mean(coco_pck_amp(y[t,:,:],true_keys[t]))
                    keyp=17
                   
                    
                            
                            
                    
                    cv2.imwrite('errors/img_'+str(t)+'.jpeg',im)
                    
                pck_blazepose_train_avg=np.mean(pck_blazepose_train)


        except:
            if True:
                print("it is not availble")



if False:
        
        # true_keys= label #np.load('data_configs/cocos_17.npy')
        
        # training data
        
        y = np.zeros((100, keyp, 3)).astype(np.uint8)
        y[0:100] = model(test_data[0:100]).numpy().astype(np.uint8)
        # y[1000:2000] = model(data[1000:2000]).numpy().astype(np.uint8)
        for t in range(100):
            tt=0
            skeleton = y[t]
            print(skeleton)
            img = test_data[t+tt].astype(np.uint8)
            for i in range(keyp):
                cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            bone(skeleton,img,keyp)
            cv2.imwrite("./results_train_aligned/coco_%d.jpg"%t, img)
            cv2.imshow("test", img)
            plt.imshow(img)
            # plt.savefig("demo_"+str(t)+".png")
            cv2.waitKey(50)
            pass
        cv2.destroyAllWindows()
        
        # testing data
        
        y = np.zeros((100,keyp, 3)).astype(np.uint8)
        test_sam1=data[data.shape[0]-101:-1]
        # test_sam_rot=test_sam[-1,:,:,:]
        # img_rotate_90_counterclockwise = cv2.rotate(test_sam_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # test_sam[-1,:,:,:]=img_rotate_90_counterclockwise
        y[0:100] = model(test_sam1).numpy().astype(np.uint8)
        
        # y[1000:2000] = model(data[1000:2000]).numpy().astype(np.uint8)
        for t in range(100):
            tt=0
            skeleton = y[t]
            print(skeleton)
            img = data[data.shape[0]-101+t].astype(np.uint8)
            for i in range(keyp):
                cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            bone(skeleton,img,keyp)
            cv2.imwrite("./result_test_aligned/coco_%d.jpg"%t, img)
            cv2.imshow("test", img)
            # plt.savefig("demo_"+str(t)+".png")
            cv2.waitKey(50)
            pass
        cv2.destroyAllWindows()
        

if False:
    for t in range(1500,0,-1):
        try:
            if pck_blazepose_avg[t-1]==0:
               pck_blazepose_avg[t-1]=pck_blazepose_avg[t]
        except:
            print('')       
            
            
if False:
    import matplotlib.pyplot as plt
    img = data[t+test_st_ind].astype(np.uint8)
    # skeleton = y[t]
    # for i in range(keyp):
    #     cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    skeleton = true_keys[t+test_st_ind].astype(int)
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=1, color=(255, 0, 0), thickness=2)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    skeleton =  ex_labels[t,:,:].astype(int)*2
    skeleton[:,0] =  skeleton[:,0]
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=1, color=(0, 255, 0), thickness=2)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    
if False:
    import matplotlib.pyplot as plt
    img = data[t].astype(np.uint8)
    test_sam2=data[t:t+1,:,:,:]
    yy = model(test_sam2).numpy().astype(np.uint8)

    skeleton = y[t]
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2].astype(int)), radius=2, color=(0, 255, 0), thickness=2)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    skeleton = true_keys[t].astype(int)
    for i in range(keyp):
        cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=1, color=(255, 0, 0), thickness=2)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
        
        # plt.axis('off')
        # plt.imshow(crop_img)
        # plt.show()save_cond=0
    
    
    # for images, labels in finetune_validation.take(1):  # only take first element of dataset
    #     numpy_images = images.numpy()
    #     numpy_labels = labels.numpy()

        
    
    
        
        
    
            