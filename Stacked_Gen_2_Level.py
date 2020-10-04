#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:10:22 2020

@author: Ogbomo-Harmitt
"""

import numpy as np


class Stacked_Gen_2_Level:
    
    def __init__(self,sub_models,Agg_1,Agg_2,agg_1_type):
        
        self.agg_1_type = agg_1_type
        Stacked_Gen_2_Level.Agg_1 = Agg_1
        Stacked_Gen_2_Level.Agg_2 = Agg_2
        
        Stacked_Gen_2_Level.sub_models = sub_models
        Stacked_Gen_2_Level.Best_sub_models = None
        
    def train(self,Features,Labels):
        
        # Training Level One Aggregator 
        
        Num_of_preds = Stacked_Gen_2_Level.sub_models.shape[0]*Stacked_Gen_2_Level.sub_models.shape[1]
        
        Sub_model_preds = np.zeros((len(Features[0]),Num_of_preds))
        
        counter = 0 
        
        for Sub_Index,Feature in enumerate(Features):
            
            Sub_Models =  Stacked_Gen_2_Level.sub_models[Sub_Index]
            
            for Sub_Model in Sub_Models:
            
                Sub_Model.fit(Feature,Labels)
                Sub_model_preds[:,counter] = Sub_Model.predict(Feature)
                    
                counter += 1 
                
        Stacked_Gen_2_Level.Agg_1.fit(Sub_model_preds,Labels)
        
        Agg_1_FI = None
        
        if self.agg_1_type == 'tree':
            
            Agg_1_FI = Stacked_Gen_2_Level.Agg_1.feature_importances_
            
            
        elif self.agg_1_type == 'linear':
            
            Agg_1_FI = Stacked_Gen_2_Level.Agg_1.coef_
        
        
        num_dif_subs = Stacked_Gen_2_Level.sub_models.shape[1]
        
        array = np.arange(num_dif_subs,Num_of_preds+num_dif_subs,num_dif_subs)
        
        best_subs_indices = []
        
        best_pred_indices = []
        
        prev_index = 0
        
        for steps,index in enumerate(array):
    
            current_FI = Agg_1_FI[prev_index:index]
            
            Max_FI_Index = np.argmax(current_FI)
            
            best_subs_indices.append(Max_FI_Index)
            
            best_pred_indices.append(Max_FI_Index + steps*num_dif_subs)
            
            prev_index = index
            
        print(best_subs_indices)
        
        temp = []
        
        for feature_Index,sub_index in enumerate(best_subs_indices):
            
            temp.append(Stacked_Gen_2_Level.sub_models[feature_Index][sub_index])
            
        Stacked_Gen_2_Level.Best_sub_models = temp
        
        # Training Level two Aggregator
        
        Best_sub_model_preds = Sub_model_preds[:,best_pred_indices]
        
        Stacked_Gen_2_Level.Agg_2.fit(Best_sub_model_preds,Labels)
            
        print("TRAINING COMPLETE")
        
        return 

    
    
    def predict(self,Features):
        
        Num_of_preds = len(Stacked_Gen_2_Level.Best_sub_models)
        
        Sub_model_preds = np.zeros((len(Features[0]),Num_of_preds))
        
        
        for Sub_Index,Feature in enumerate(Features):
            
            Sub_Model =  Stacked_Gen_2_Level.Best_sub_models[Sub_Index]

            Sub_model_preds[:,Sub_Index] = Sub_Model.predict(Feature)
            
        
        Final_Preds = Stacked_Gen_2_Level.Agg_2.predict(Sub_model_preds)
        
        return Final_Preds
    
    
    def Get_Feature_Importances(self):
        
        FI_array = []
        
        for model in Stacked_Gen_2_Level.Best_sub_models:
            
            current_FI = model.feature_importances_
            
            FI_array.append(current_FI)
            
        return FI_array