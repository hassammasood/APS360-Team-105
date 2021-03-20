# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:17:59 2021

@author: Roméo Cabrera & Suleman Qamar

"""
'''Cell 0: Importing Libraries'''

import numpy as np
import os
import glob
import re
import time
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import cv2
import imutils
from torchvision.datasets.folder import default_loader
#%%
'''Cell 1: Converting Train & Validation Images to .jpg'''
### This must be done in order to set-up the baseline model using sklearn, ###
### as it will be much easier to load in the images if they are all of the same type. ###

train_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\train'
validation_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\val'

ext_list=['jpg','jpeg','png']
for ext in ext_list:
    if ext_list.index(ext)==0:
        train_fname_list_positive=[fname for fname in glob.glob(train_BD_path+'\\1\\'+f'*.{ext}')]
        train_fname_list_negative=[fname for fname in glob.glob(train_BD_path+'\\2\\'+f'*.{ext}')]
        validation_fname_list_positive=[fname for fname in glob.glob(validation_BD_path+'\\1\\'+f'*.{ext}')]
        validation_fname_list_negative=[fname for fname in glob.glob(validation_BD_path+'\\2\\'+f'*.{ext}')]
    else:
        train_fname_list_positive+=[fname for fname in glob.glob(train_BD_path+'\\1\\'+f'*.{ext}')]
        train_fname_list_negative+=[fname for fname in glob.glob(train_BD_path+'\\2\\'+f'*.{ext}')]
        validation_fname_list_positive+=[fname for fname in glob.glob(validation_BD_path+'\\1\\'+f'*.{ext}')]
        validation_fname_list_negative+=[fname for fname in glob.glob(validation_BD_path+'\\2\\'+f'*.{ext}')]

# indexlist=[train_fname_list_positive.index(png) for png in train_fname_list_positive if '.png' in png]


### Add together the positive and negative train and validation data to get the total
train_list=train_fname_list_negative+train_fname_list_positive
validation_list=validation_fname_list_negative+validation_fname_list_positive

# Rename all of the files as jpg files, removing jpeg and png types.
# Some png/jpeg files have the same name as existing jpg files, need to handle
count=0
for file in train_list:
    fname_length=len(os.path.splitext(file)[0].split('\\')[-1])
    if file.split('.')[-1]!='jpg':
        if os.path.splitext(file)[0]+'.jpg' not in train_list:
            os.rename(file,os.path.splitext(file)[0]+'.jpg')
        else:
            if 'Y' in file:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'NewNameY{count}'+'.jpg')
            else:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'New{count} no'+'.jpg')
    count+=1

count=0
for file in validation_list:
    fname_length=len(os.path.splitext(file)[0].split('\\')[-1])
    if file.split('.')[-1]!='jpg':
        if os.path.splitext(file)[0]+'.jpg' not in validation_list:
            os.rename(file,os.path.splitext(file)[0]+'.jpg')
        else:
            if 'Y' in file:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'NewNameY{count}'+'.jpg')
            else:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'New{count} no'+'.jpg')
    count+=1
#%%
'''
def brain_contours(image, plots=False): 
    # blur the image 
    #image = np.array(image)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(image, (6,6), 0)
    #Erosion 
    #kernel = np.ones((5,5), np.uint8)  
    #eroded = cv2.erode(image, kernel, iterations = 1)
    #Dilate the image 
    #dilated = cv2.dilate(eroded, kernel, iterations= 1)
    #identify the contours 
    contours= cv2.findContours(image,  cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_EXTERNAL)
    contours = max(imutils.grab_contours(contours))
    top , bottom , left , right = tuple(countours[countours[:, :, 1].argmax[0]]), tuple(countours[countours[:, :, 1].argmin[0]]), tuple(countours[countours[:, :, 0].argmin[0]]), tuple(countours[countours[:, :, 0].argmax[0]])
    image = image[top[1] : bottom[1], right[1] : left [1]]
    return image
'''
                                                        
#%%
#Cell 2a: Importing Data 
train_BD_path='C:\\Users\\Surface\\Documents\\APS360\\archive\\Tumeur_cerveau\\train'
validation_BD_path='C:\\Users\\Surface\\Documents\\APS360\\archive\\Tumeur_cerveau\\val'
'''
#transform_group=transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(), 
                                    transforms.ToTensor()])
train_data_BD=ImageFolder(root=train_BD_path,transform=transform_group)
validation_data_BD=ImageFolder(root=validation_BD_path,transform=transform_group)'''

#%%
class CroppedData(ImageFolder): 
    def __init__(self, root): 
        super(CroppedData, self).__init__(root = root, loader = default_loader, is_valid_file = None)
    
    def __getitem__(self, index): 
        path , target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(image)
        contours = cv2.findContours(image,  cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST)
        contours = imutils.grab_contours(contours)
        contours = max(contours, key=cv2.contourArea)
        top = tuple(contours[contours[:, :, 1].argmin()][0])
        bottom = tuple(contours[contours[:, :, 1].argmax()][0])
        left =  tuple(contours[contours[:, :, 0].argmin()][0])
        right = tuple(contours[contours[:, :, 0].argmax()][0])
        print(top, bottom, right, left)
        cropped_image = image[top[1]:bottom[1], left[0]:right[0]] 
        resized_image = cv2.resize(cropped_image, (224, 224))
        output_image = torchvision.transforms.ToTensor()(cropped_image)
        return output_image, target
        
data_train = CroppedData(train_BD_path)
data_val = CroppedData(validation_BD_path)
train_loader = torch.utils.data.DataLoader(data_train, batch_size = 1)
val_loader = torch.utils.data.DataLoader(data_val, batch_size = 1)

#%%
'''Cell 2b: Visualize Subset of Train Data'''
#train_loader=DataLoader(train_data_BD,batch_size=1,shuffle=True)
k = 0
for images, labels in train_loader:
    # since batch_size = 1, there is only 1 image in `images`
    # print(image.shape)
    image = images[0]
    #image = brain_countours(image , plots = False )
    # print(images.shape)
    # place the colour channel at the end, instead of at the beginning
    img = np.transpose(image, [1,2,0])
    # normalize pixel intensity values to [0, 1]
    # img = img / 2 + 0.5
    plt.subplot(3, 5, k+1)
    plt.axis('off')
    plt.imshow(img, cmap= 'gray')

    k += 1
    if k > 14:
        break
#%%