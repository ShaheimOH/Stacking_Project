#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:00:06 2020

@author: Shaheim Ogbomo-Harmitt

K-FOLD CROSS-VALIDATION MODULE FOR STACKED GENERALISATION MODEL

This module performs a K-Fold cross-validation for a stacking mode 
(with a K = 5). The module outputs mean absolute error of all the folds for
regression and the accuracy score for classification. 

Inputs:
    
    Stack_Model - Object of stacking model
    
    Features - Python list of features for each modalities.
    
    Labels - Phenotypes of subjects of dataset.
    
    Type -  Type of prediction (Regression or Classification)."Reg","Class".
    
    
Outputs:
    
    Run() - This performs the K-Fold cross validation and outputs the error.
    
    Get_FI() - Returns average importance of each feature from the 5 folds.
    


"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

 
class K_Fold_Stack():
    
    def __init__(self,Stack_Model,Features,Labels,Type):
        
        self.Stack_Model = Stack_Model
        self.Features = Features
        self.Labels = Labels
        self.Type = Type
        K_Fold_Stack.FI_array = None
        
        
    def train_test_feats_stack(self,indices_test,indices_train,Features_Array):
    
        test_features = []
        train_features = []
    
        for features in Features_Array:
            test_feats =  features[indices_test,:]
            train_feats =  features[indices_train,:]
        
            test_features.append(test_feats)
            train_features.append(train_feats)
        
        
        return train_features,test_features
    
    
    def Add_Importances(self,Array_1,Array_2):
    
        new_FI = []
    
        for mod_index,mod_1 in enumerate(Array_1):
        
           mod_2 = Array_2[mod_index]
        
           new_FI.append(np.add(mod_1,mod_2))
        
        return new_FI
    

    def Mean_FI(self,Importances_Array):
    
        FI_0 = Importances_Array[0]
    
        array = [1,2,3,4]
    
        for i in array:
            
            current_FI = Importances_Array[i]
            FI_0 = self.Add_Importances(FI_0,current_FI)
            
            
        for i in range(len(FI_0)):
            
            FI_0[i] = FI_0[i]/5
        
        return FI_0
    
    
    def Run(self):
    
        kf = KFold(n_splits=5)
        
        error = []
        
        FI_array = []

        for train_index, test_index in kf.split(self.Features[0]):
            
            X_train, X_test = self.train_test_feats_stack(test_index,train_index,self.Features)
            y_train, y_test = self.Labels[train_index], self.Labels[test_index]
            Stack_Obj = self.Stack_Model
            Stack_Obj.train(X_train,y_train)
            pred = Stack_Obj.predict(X_test)
            FI = Stack_Obj.Get_Feature_Importances()
            FI_array.append(FI)
            
            if self.Type == 'Reg':
                
                error.append(mean_absolute_error(y_test,pred))
                
            if self.Type == 'Class':
                
                error.append(accuracy_score(y_test,pred))
                
        
        K_Fold_Stack.FI_array = FI_array
        
        return np.mean(error)
    
    
    def Get_FI(self):
        
        return self.Mean_FI(K_Fold_Stack.FI_array)
        