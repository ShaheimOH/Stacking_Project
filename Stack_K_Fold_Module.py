#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:00:06 2020

@author: Shaheim Ogbomo-Harmitt
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

class K_Fold_Stack():
    
    def __init__(self,Stack_Model,Features,Labels):
        
        self.Stack_Model = Stack_Model
        self.Features = Features
        self.Labels = Labels
        
        
    def train_test_feats_stack(self,indices_test,indices_train,Features_Array):
    
        test_features = []
        train_features = []
    
        for features in Features_Array:
            test_feats =  features[indices_test,:]
            train_feats =  features[indices_train,:]
        
            test_features.append(test_feats)
            train_features.append(train_feats)
        
        
        return train_features,test_features


    def Run(self):
    
        kf = KFold(n_splits=5)
        
        error = []

        for train_index, test_index in kf.split(self.Features[0]):
            
            X_train, X_test = self.train_test_feats_stack(test_index,train_index,self.Features)
            y_train, y_test = self.Labels[train_index], self.Labels[test_index]
            Stack_Obj = self.Stack_Model
            Stack_Obj.train(X_train,y_train)
            pred = Stack_Obj.predict(X_test)
            error.append(mean_absolute_error(y_test,pred))
        
        return np.mean(error)