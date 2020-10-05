
"""
Created on Wed Aug 19 16:10:22 2020

@author: Shaheim Ogbomo-Harmitt

STACKED GENERALISATION TWO-LEVEL MODEL

This module is two level stacking model for ML models within the Sklearn 
library.

Inputs:
    
    sub_models - Python list of ML models for stacking.
    
    Agg_1 - First aggregator of model, the one which picks the best sub-model
            for each modality.
    
    Agg_2 - Second aggregator of model, combines predictions of all the
            modalities.
    
    agg_1_type - Type of ML for first aggregator. 'tree','linear'.
    
    agg_2_type - Type of ML for second aggregator. 'tree','linear'.
    

Operations:
    
    Train() - This trains the stacking model.

    
Outputs:
    
    Predict() - Predicts phenotypes from an independent feature sets. Outputs 
              Numpy array of predictions.
              
    Get_Feature_Importances() - Outputs a list with arrays of feature 
                                importances of each sub model.


"""

import numpy as np

class Stacked_Gen_2_Level:
    
    def __init__(self,sub_models,Agg_1,Agg_2,agg_1_type,agg_2_type):
        
        self.agg_1_type = agg_1_type
        self.agg_2_type = agg_2_type
        Stacked_Gen_2_Level.Agg_1 = Agg_1
        Stacked_Gen_2_Level.Agg_2 = Agg_2
        
        Stacked_Gen_2_Level.sub_models = sub_models
        Stacked_Gen_2_Level.Best_sub_models = None
    
    #TRAINING
    
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
        
        temp = []
        
        for feature_Index,sub_index in enumerate(best_subs_indices):
            
            temp.append(Stacked_Gen_2_Level.sub_models[feature_Index][sub_index])
            
        Stacked_Gen_2_Level.Best_sub_models = temp
        
        # Training Level two Aggregator
        
        Best_sub_model_preds = Sub_model_preds[:,best_pred_indices]
        
        Stacked_Gen_2_Level.Agg_2.fit(Best_sub_model_preds,Labels)
            
        print("TRAINING COMPLETE")
        
        return 


    # PREDICTION
    
    def predict(self,Features):
        
        Num_of_preds = len(Stacked_Gen_2_Level.Best_sub_models)
        
        Sub_model_preds = np.zeros((len(Features[0]),Num_of_preds))
        
        
        for Sub_Index,Feature in enumerate(Features):
            
            Sub_Model =  Stacked_Gen_2_Level.Best_sub_models[Sub_Index]

            Sub_model_preds[:,Sub_Index] = Sub_Model.predict(Feature)
            
        
        Final_Preds = Stacked_Gen_2_Level.Agg_2.predict(Sub_model_preds)
        
        return Final_Preds
    
    
    # FEATURE IMPORTANCE 
    
    def Get_Feature_Importances(self):
        
        FI_array = []
        
        for index,model in enumerate(Stacked_Gen_2_Level.Best_sub_models):
            
            if self.agg_2_type == 'tree':
            
                Mod_FI = Stacked_Gen_2_Level.Agg_2.feature_importances_[index]
            
            elif self.agg_2_type == 'linear':
            
                Mod_FI = Stacked_Gen_2_Level.Agg_2.coef_[index]
            
            current_FI = model.feature_importances_
            
            current_FI = Mod_FI*current_FI
            
            current_FI = current_FI/np.amax(current_FI)
            
            FI_array.append(current_FI)
            
        return FI_array