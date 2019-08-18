#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:45 2019

@author: sameer
"""

import torch
import numpy as np
from scipy.spatial import distance

(x_train, y_train, x_test, y_test)=torch.load('mnist.pt')
a,b,c=x_train.shape
train=[]
for i in x_train:
    temp=i.view(1,-1)[0]
    train.append(temp)
a1,b1,c1=x_test.shape
test=[]
for j in x_test:
    tmp=j.view(1,-1)[0]
    test.append(tmp)

Dist=torch.zeros([100, 1000])
p1=x_train[0]
for i in range(a1):
    for j in range(a):
        aa=distance.euclidean(train[j].numpy(),test[i].numpy())
        Dist[i,j]=aa

def KNN(k,Dist,y_train,y_test):
    match=0
    aa,bb=Dist.shape
    sort_dist=torch.zeros([aa,bb])
    print(type(sort_dist))
    for i in range(aa):
        sort_dist[i,:],dummy=torch.sort(Dist[i,:])
    min_K=sort_dist[:,0:k]
    a1,b1=min_K.shape
    temp2=torch.zeros([a1,b1])
    result=[]
    for i in range(a1):
        count=0
        for j in range(b1):
            for k1 in range(bb):
                if Dist[i,k1]==min_K[i,j]:
                    temp2[i,j]=y_train[k1]
                    break
    
    for i in range(a1):
        count=0
        for j in range(b1):
            if temp2[i,j]==y_test[i].item():
                count=count+1
        if count>=((k+1)/2):
            match=match+1
            result.append(int(y_test[i].item()))
        else:
            result.append(int(temp2[i,0].item()))
    return (match,result)
(x,result)=KNN(3,Dist,y_train,y_test)

print(list(y_test.numpy()))
print(result)

                        
                    
 

           
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


class_names = ['0','1','2','3','4','5','6','7','8','9']




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true,y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test,np.array(result), classes=class_names,
                      title='Confusion matrix, without normalization')


plt.savefig("conf.png")
plt.show()            
            
            
            
            
            
            
            
        

        
        
    