# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:49:54 2021

@author: Roméo Cabrera & Viktor Kornyeyev
"""

'''Cell 0: Importing Libraries'''

import numpy as np
import os
import glob
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
use_cuda=False
#%%
'''Cell 1: Defining CNN'''
class TumourClassifier(nn.Module):
    def __init__(self):
      super(TumourClassifier, self).__init__()
      self.name = "TumourBinary"
      self.conv1 = nn.Conv2d(1, 5, 3) # in_channels, out_channels, kernel_size, stride, padding
      self.pool  = nn.MaxPool2d(2, 2) # kernel_size, stride
      self.conv2 = nn.Conv2d(5, 10, 5)
      self.conv3 = nn.Conv2d(10, 5, 3)
      self.fc1 = nn.Linear(5*25*25, 36) # Number of inputs, number of outputs (neurons)
      self.fc2 = nn.Linear(36, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, 5*25*25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x
#%%
'''Cell 2a: Defining Accuracy Function'''
def get_accuracy(model, loader):
    
    correct = 0
    total = 0
    for images, labels in iter(loader):
        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
        #############################################
        pred = model(images) # We don't need to run F.softmax
        # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += images.shape[0]
    return correct / total
#%%
'''Cell 2b: Defining Validation Loss Reporting Function'''
def losses(model,loader,criterion):
  losses=0
  iterations=0
  for images,labels in iter(loader):
      #############################################
      #To Enable GPU Usage
      if use_cuda and torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()
      #############################################
      recon = model(images)
      labels=labels.type_as(recon)
      loss = criterion(recon, labels)
      losses += float(loss)
      iterations += 1
  average_loss=float(losses/iterations)
  return average_loss
#%%
'''Cell 2c: Defining Model Naming Function for Saving'''
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path
#%%
'''Cell 3: Train Function Definition'''

### Input path in which you want the model and optimizer states and statistics to be saved###
drive_dir_model= 'C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\Models'

def train_cnn(model, train_loader, validation_loader, start_epoch=0, num_epochs=5, batch_size=32, learning_rate=1e-5, weight_decay=0, interval=2, plot=True):
    
    torch.manual_seed(5)
    criterion = nn.BCEWithLogitsLoss() #Binary classification problem
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader=torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    validation_loader=torch.utils.data.DataLoader(validation_loader, batch_size=batch_size, shuffle=True)
    
    train_loss,train_accuracy,validation_loss,validation_accuracy=[],[],[],[]
    
    for epoch in range(start_epoch,num_epochs):
      epoch_start=time.time()
      train_loss_value=0
      iterations=0
      for images, labels in iter(train_loader):
          #############################################
          #To Enable GPU Usage
          if use_cuda and torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda()
          #############################################
          optimizer.zero_grad()
          pred = model(images)
          labels=labels.type_as(pred)
          loss = criterion(pred, labels)
          loss.backward()
          optimizer.step()

          iterations+=1
          train_loss_value+=float(loss)
        
      
      epoch_end=time.time()
      epoch_time=epoch_end-epoch_start
      

      train_accuracy.append(get_accuracy(model,train_loader))
      validation_accuracy.append(get_accuracy(model,validation_loader))
      train_loss.append(train_loss_value/iterations)
      validation_loss.append(losses(model,validation_loader,criterion))
      

      if epoch%interval==0 or epoch==num_epochs-1:
        print(f"Epoch {epoch+1}: Train accuracy: {train_accuracy[epoch]} |"
            +f" Validation accuracy: {validation_accuracy[epoch]} |"+f" Training Loss: {train_loss[epoch]}"
            +f" Validation Loss: {validation_loss[epoch]}")
        print('Time elapsed for epoch: {:.2f} seconds'.format(epoch_time))
     
      model_path = drive_dir_model + '\\' + get_model_name(model.name, batch_size, learning_rate, epoch, model.hidden_size)+'_weightdecay'+str(weight_decay)
        
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss,
        'train_accuracy': train_accuracy,
        'validation_accuracy': validation_accuracy
        }, model_path)

    # plotting
    epochs=[x+1 for x in range(start_epoch,num_epochs)]

    if plot:
      plt.title("Training VS Validation Loss")
      plt.plot(epochs, train_loss, label="Train")
      plt.plot(epochs, validation_loss, label="Validation")
      plt.xlabel("Epochs")
      plt.ylabel("Loss")
      plt.legend(loc='best')
      plt.show()

      plt.title("Training VS Validation Accuracy")
      plt.plot(epochs, train_accuracy, label="Train")
      plt.plot(epochs, validation_accuracy, label="Validation")
      plt.xlabel("Epochs")
      plt.ylabel("Accuracy")
      plt.legend(loc='best')
      plt.show()

    np.savetxt(f"{model_path}_train_accuracy.csv", train_accuracy)
    np.savetxt(f"{model_path}_train_loss.csv", train_loss)
    np.savetxt(f"{model_path}_validation_accuracy.csv", validation_accuracy)
    np.savetxt(f"{model_path}_validation_loss.csv", validation_loss)

    print("Best Train Accuracy: ", max(train_accuracy))
    print("Best Validation Accuracy: ", max(validation_accuracy))
    print(f"Final Training Accuracy: {train_accuracy[-1]}")
    print(f"Final Validation Accuracy: {validation_accuracy[-1]}")
    print(f"Final Training Loss: {train_loss[-1]}")
    print(f"Final Validation Loss: {validation_loss[-1]}")
#%%
'''Cell 4: Running Model'''

TumourBinary=TumourClassifier()
if torch.cuda.is_available():
  TumourBinary=TumourBinary.cuda()
  print('GPU')
print('No GPU')
train_cnn(TumourBinary,train_data_BD,validation_data_BD, num_epochs=15, batch_size=32, learning_rate=0.001, interval=3)
#%%
'''Cell 5: Model Evaluation'''
