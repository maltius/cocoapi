#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:05:30 2021

@author: altius
"""

import cv2
import os
import cv2
import numpy as np
import glob

image_folder = '/mnt/sda1/downloads/BlazePose-tensorflow-master/temp/results_hm_main/'
video_name = 'output_video_tube.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()




frameSize = (width, height)

out = cv2.VideoWriter(image_folder+video_name,cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize)

for filename in sorted(glob.glob(image_folder+'*.jpg')):
    img = cv2.imread(filename)
    # img2=cv2.resize(img,frameSize)
    # print(img.shape)
    # print(filename)
    out.write(img)

out.release()