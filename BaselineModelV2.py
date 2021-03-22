# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:27:55 2021

@author: Roméo Cabrera & Hassam Masood
"""
'''Cell 0: Importing Libraries'''
import numpy as np
import pandas as pd
import os
import time
import glob
from PIL import Image
import sklearn as sk
from sklearn import svm, metrics
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
#%%
'''Cell 1: Loading in Data and Preprocessing for SKLearn'''

train_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\train'
test_BD_path='C:\\Users\\Home\\Desktop\\Roméo\\Courses\\APS360 Applied Fundamentals Of Machine Learning\\Project\\BD_Brain-Tumor\\Tumeur_cerveau\\test'

train_fname_list_positive=[fname for fname in glob.glob(train_BD_path+'\\1\\'+'*.jpg')]
train_fname_list_negative=[fname for fname in glob.glob(train_BD_path+'\\2\\'+'*.jpg')]

test_fname_list_positive=[fname for fname in glob.glob(test_BD_path+'\\1\\'+'*.jpg')]
test_fname_list_negative=[fname for fname in glob.glob(test_BD_path+'\\2\\'+'*.jpg')]


train_positive_data = pd.DataFrame(columns=(['img_name','label']))
train_negative_data = pd.DataFrame(columns=(['img_name','label']))

test_positive_data = pd.DataFrame(columns=(['img_name','label']))
test_negative_data = pd.DataFrame(columns=(['img_name','label']))

train_positive_labels = np.ones(len(train_fname_list_positive))
train_negative_lables = np.zeros(len(train_fname_list_negative))

test_positive_labels = np.ones(len(test_fname_list_positive))
test_negative_lables = np.zeros(len(test_fname_list_negative))

train_positive_data['img_name'] = train_fname_list_positive
train_negative_data['img_name'] = train_fname_list_negative
train_positive_data['label'] = train_positive_labels
train_negative_data['label'] = train_negative_lables

test_positive_data['img_name'] = test_fname_list_positive
test_negative_data['img_name'] = test_fname_list_negative
test_positive_data['label'] = test_positive_labels
test_negative_data['label'] = test_negative_lables

train_data = pd.concat([train_positive_data,train_negative_data] , axis =0)
train_data = sk.utils.shuffle(train_data, random_state = 5)

test_data = pd.concat([test_positive_data,test_negative_data] , axis =0)
test_data = sk.utils.shuffle(test_data, random_state = 5)

### Checking to make sure it works ###
train_data_check=train_data.iloc[:100,:]
len(train_data_check[train_data_check['label']==1])

test_data_check=test_data.iloc[:100,:]
len(test_data_check[test_data_check['label']==1])
#%%
'''Cell 2: Extracting Features From Images'''
def get_image(row):
    """
    Converts an image number into the file path where the image is located, 
    opens the image, and returns the image as a numpy array.
    """
    file_path = row
    img = Image.open(file_path)
    img=img.resize((224,224))
    return np.array(img)


def create_features(img):
    # flatten three channel color image
    # color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image)#, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    # flat_features = np.hstack(color_features)
    return hog_features


def create_feature_matrix(images_df):
    features_list = []
    
    for img_id in images_df.img_name:
        # load image
        img = get_image(img_id)
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)#,dtype='object')
    return feature_matrix

# run create_feature_matrix on our dataframe of images
train_feature_matrix = create_feature_matrix(train_data)
test_feature_matrix = create_feature_matrix(test_data)
#%%
'''Cell 3: Reduce and Scale Feature Matrices to Embeddings'''
# get shape of feature matrix
print('Train feature matrix shape is: ', train_feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
train_feature_matrix_standard = ss.fit_transform(train_feature_matrix)
test_feature_matrix_standard = ss.fit_transform(test_feature_matrix)

pca = PCA(n_components=50)
# use fit_transform to run PCA on our standardized matrix
X_train = pca.fit_transform(train_feature_matrix_standard)
X_test = pca.fit_transform(test_feature_matrix_standard)

y_train=train_data['label'].values
y_test=test_data['label'].values
# look at new shape
print('PCA matrix shape is: ', X_train.shape)
#%%
'''Cell 4: Define Train/Predict Function'''
def train_and_predict(train_input_features, train_outputs, prediction_features):
    """
    :param train_input_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width   
    :param train_outputs: (numpy.array) A one-dimensional NumPy array where each element
                        is a number representing the species of iris which is described in
                        the same row of train_input_features. 0 represents Iris setosa,
                        1 represents Iris versicolor, and 2 represents Iris virginica.
    :param prediction_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width
    :returns: (list) The function should return an iterable (like list or numpy.ndarray) of the predicted 
                        iris species, one for each item in prediction_features
    """   
    
    classif=svm.SVC(kernel='poly',decision_function_shape='ovo',random_state=0)
    classif.fit(train_input_features,train_outputs)
    
    output_class=np.array(classif.predict(prediction_features))
    
    return output_class


#%%
'''Cell 5: Train and Evaluate model on Test'''


y_pred = train_and_predict(X_train, y_train , X_test)
if y_pred is not None:
    print(metrics.accuracy_score(y_test, y_pred))

#%%
'''Cell 6: ROC Curve, AUC, Confusion Matrix'''
# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');