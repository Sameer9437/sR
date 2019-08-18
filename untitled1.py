#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:49:56 2019

@author: sameer
"""

import torch
import numpy as np
from scipy.spatial import distance

(x_train, y_train, x_test, y_test)=torch.load('cifar10.pt')
histogram_train=[]
a,b,c,d=x_train.shape
a1,b1,c1,d1=x_test.shape
for i in range(a):
    temp=x_train[i]
    a2,b2,c2=temp.shape
    hist_r=torch.zeros([1,256])
    hist_g=torch.zeros([1,256])
    hist_b=torch.zeros([1,256])
    for j in range(c2):
        count=0
        for k in range(a2):
            for l in range(b2):
                if j<1:
                    value=temp[k,l,j].item()
                    hist_r[0,value]=hist_r[0,value]+1
                    count=count+1
                elif j<2:
                    value=temp[k,l,j]
                    hist_g[0,value]=hist_g[0,value]+1
                    count=count+1
                else:
                    value=temp[k,l,j]
                    hist_b[0,value]=hist_b[0,value]+1
                    count=count+1
#    hst=list(hist_r.numpy())+list(hist_g.numpy())+list(hist_b.numpy())
#    hst=torch.from_numpy(hst[0])
    aa=hist_r.numpy()
    bb=hist_g.numpy()
    cc=hist_b.numpy()
                
    hst=np.concatenate((aa,bb,cc),axis=1)
    hs=torch.from_numpy(hst)                
    histogram_train.append(hs)

histogram_test=[]
for i in range(a1):
    temp1=x_test[i]
    a3,b3,c3=temp1.shape
    hist_r1=torch.zeros([1,256])
    hist_g1=torch.zeros([1,256])
    hist_b1=torch.zeros([1,256])
    for j in range(c3):
        count=0
        for k in range(a3):
            for l in range(b3):
                if j<1:
                    value=temp1[k,l,j].item()
                    hist_r1[0,value]=hist_r1[0,value]+1
                elif j<2:
                    value=temp1[k,l,j]
                    hist_g1[0,value]=hist_g1[0,value]+1
                else:
                    value=temp[k,l,j]
                    hist_b1[0,value]=hist_b1[0,value]+1
#    hst=list(hist_r.numpy())+list(hist_g.numpy())+list(hist_b.numpy())
#    hst=torch.from_numpy(hst[0])
    aa1=hist_r1.numpy()
    bb1=hist_g1.numpy()
    cc1=hist_b1.numpy()
                
    hst1=np.concatenate((aa1,bb1,cc1),axis=1) 
    hs1=torch.from_numpy(hst1)               
    histogram_test.append(hs1)
               
                
Dist=torch.zeros([100, 1000])

for i in range(a1):
    for j in range(a):
        aa1=distance.euclidean(histogram_train[j],histogram_test[i])
        Dist[i,j]=aa1

def KNN(k,Dist,y_train,y_test):
    aa,bb=Dist.shape
    sort_dist=torch.zeros([aa,bb])
    for i in range(aa):
        sort_dist[i,:],dummy=torch.sort(Dist[i,:])
    min_K=sort_dist[:,0:k]
    a1,b1=min_K.shape
    temp2=torch.zeros([a1,b1])
    result=[]
    for i in range(a1):
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
            result.append(int(y_test[i].item()))
        else:
            result.append(int(temp2[i,0].item()))
    return result
result=KNN(3,Dist,y_train,y_test)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
