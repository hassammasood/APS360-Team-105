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
#%%
'''Cell 0b: Converting Train & Validation & Test Images to .jpg'''
### This must be done in order to set-up the baseline model using sklearn, ###
### as it will be much easier to load in the images if they are all of the same type. ###

train_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\train'
validation_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\val'
test_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\test'

ext_list=['jpg','jpeg','png']
for ext in ext_list:
    if ext_list.index(ext)==0:
        train_fname_list_positive=[fname for fname in glob.glob(train_BD_path+'\\1\\'+f'*.{ext}')]
        train_fname_list_negative=[fname for fname in glob.glob(train_BD_path+'\\2\\'+f'*.{ext}')]
        validation_fname_list_positive=[fname for fname in glob.glob(validation_BD_path+'\\1\\'+f'*.{ext}')]
        validation_fname_list_negative=[fname for fname in glob.glob(validation_BD_path+'\\2\\'+f'*.{ext}')]
        test_fname_list_positive=[fname for fname in glob.glob(test_BD_path+'\\1\\'+f'*.{ext}')]
        test_fname_list_negative=[fname for fname in glob.glob(test_BD_path+'\\2\\'+f'*.{ext}')]
    else:
        train_fname_list_positive+=[fname for fname in glob.glob(train_BD_path+'\\1\\'+f'*.{ext}')]
        train_fname_list_negative+=[fname for fname in glob.glob(train_BD_path+'\\2\\'+f'*.{ext}')]
        validation_fname_list_positive+=[fname for fname in glob.glob(validation_BD_path+'\\1\\'+f'*.{ext}')]
        validation_fname_list_negative+=[fname for fname in glob.glob(validation_BD_path+'\\2\\'+f'*.{ext}')]
        test_fname_list_positive+=[fname for fname in glob.glob(test_BD_path+'\\1\\'+f'*.{ext}')]
        test_fname_list_negative+=[fname for fname in glob.glob(test_BD_path+'\\2\\'+f'*.{ext}')]

# indexlist=[train_fname_list_positive.index(png) for png in train_fname_list_positive if '.png' in png]


### Add together the positive and negative train and validation data to get the total
train_list=train_fname_list_negative+train_fname_list_positive
validation_list=validation_fname_list_negative+validation_fname_list_positive
test_list=test_fname_list_negative+test_fname_list_positive

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

count=0
for file in test_list:
    fname_length=len(os.path.splitext(file)[0].split('\\')[-1])
    if file.split('.')[-1]!='jpg':
        if os.path.splitext(file)[0]+'.jpg' not in test_list:
            os.rename(file,os.path.splitext(file)[0]+'.jpg')
        else:
            if 'Y' in file:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'NewNameY{count}'+'.jpg')
            else:
                os.rename(file,os.path.splitext(file)[0][:-fname_length]+f'New{count} '+'no'+'.jpg')
    count+=1
                                                            
#%%
'''Cell 1a: Importing Data'''
train_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\train'
validation_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\val'
test_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\test'

transform_group=transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(), 
                                    transforms.ToTensor()])
train_data_BD=ImageFolder(root=train_BD_path,transform=transform_group)
validation_data_BD=ImageFolder(root=validation_BD_path,transform=transform_group)

#%%
'''Cell 1b: Visualize Subset of Train Data'''
train_loader=DataLoader(train_data_BD,batch_size=1,shuffle=True)
k = 0
for images, labels in train_loader:
    # since batch_size = 1, there is only 1 image in `images`
    # print(image.shape)
    image = images[0]
    # print(images.shape)
    # place the colour channel at the end, instead of at the beginning
    img = np.transpose(image, [1,2,0])
    # normalize pixel intensity values to [0, 1]
    # img = img / 2 + 0.5
    plt.subplot(3, 5, k+1)
    plt.axis('off')
    plt.imshow(img,cmap='gray')

    k += 1
    if k > 14:
        break
#%%