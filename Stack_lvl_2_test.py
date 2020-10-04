#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:42:15 2020

@author: Ogbomo-Harmitt
"""

## DATA IMPORTING 

import pandas as pd
import numpy as np
from Data_Processing_Ver_1_1 import *
from Single_Model_Exp_Module import *
from Stacked_Gen_2_Level import *
from Stacked_Gen import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def Fix_Data(Data):
    
    New_Data = np.delete(Data, [0,83], 1)
    
    return New_Data

# DATA PREPROCESSING 

T1_T2_All_Info = pd.read_csv("T1_T2_info.csv")
Myelin_All_Info = pd.read_csv("Myelin_info_LR.csv")
Curvature_All_Info = pd.read_csv("Curvature_info_LR.csv")
Sulc_All_Info = pd.read_csv("Sulc_info_LR.csv")
SC_All_Info = pd.read_csv("SC_Full_IDs.csv")
Vols_All_Info = pd.read_csv("Vols_info.csv")

T1_T2_Full_IDs = T1_T2_All_Info.iloc[:,1].tolist()
Myelin_Full_IDs = Myelin_All_Info.iloc[:,1].tolist()
Curvature_Full_IDs = Curvature_All_Info.iloc[:,1].tolist()
Sulc_Full_IDs = Sulc_All_Info.iloc[:,1].tolist()
SC_Full_IDs = SC_All_Info.iloc[:,1].tolist()
Vols_Full_IDs = Vols_All_Info.iloc[:,1].tolist()
Diff_Full_IDs = pd.read_csv("Diff_Full_IDs.csv").iloc[:,1].tolist()

T1_T2_Features = pd.read_csv("T1_T2_data_v2.csv").iloc[:,1:].values
T1_T2_Features = Fix_Data(T1_T2_Features)
Myelin_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withmyelin_rois_LR.pk1").iloc[:,:-1].values
Curvature_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withcurvature_rois_LR.pk1").iloc[:,:-1].values
Sulc_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withsulc_rois_LR.pk1").iloc[:,:-1].values
SC_Features = pd.read_csv("SC_FS_Birth_Features.csv").iloc[:,1:].values
Vols_Features = pd.read_csv("Vol_data_v2.csv").iloc[:,1:].values
Vols_Features = Fix_Data(Vols_Features)
Diff_Features = pd.read_csv("Diff_Data_V1.csv").iloc[:,1:].values
Diff_Features = Fix_Data(Diff_Features)

# Phenotypes 

Phenotypes_Info = pd.read_excel("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga.xlsx")
Phenotypes_IDs = Phenotypes_Info.iloc[:,0].tolist()
GA_at_Birth = Phenotypes_Info.iloc[:,10].values
GA_at_Scan = Phenotypes_Info.iloc[:,11].values
scanner_validation = Phenotypes_Info.iloc[:,4].values

Phenotypes_IDs_Full = []

for k,ID in enumerate(Phenotypes_IDs):
    
    full_ID = 'sub-' + ID + '_ses-' + str(scanner_validation[k])
    full_ID = full_ID[:-2]
    
    Phenotypes_IDs_Full.append(full_ID)
    
    
# Processing 
    
IDs_Array = [Myelin_Full_IDs,Curvature_Full_IDs,Sulc_Full_IDs,T1_T2_Full_IDs,SC_Full_IDs,Vols_Full_IDs,Diff_Full_IDs]
Data_Processing_Obj = Data_Processing(IDs_Array,GA_at_Birth,Phenotypes_IDs_Full)
Filtered_Indices = Data_Processing_Obj.Get_Indices()
Labels_Fil = Data_Processing_Obj.Get_Labels()

T1_T2_Features_Fil = T1_T2_Features[Filtered_Indices[:,3],:]
Myelin_Features_Fil = Myelin_Features[Filtered_Indices[:,0],:]
Curvature_Features_Fil = Curvature_Features[Filtered_Indices[:,1],:]
Sulc_Features_Fil = Sulc_Features[Filtered_Indices[:,2],:]
SC_Features_Fil = SC_Features[Filtered_Indices[:,4],:]
Vols_Features_Fil = Vols_Features[Filtered_Indices[:,5],:]


# Zero/Mean Filled 

Features = [Myelin_Features,Curvature_Features,Sulc_Features,T1_T2_Features,SC_Features,Vols_Features,Diff_Features]

Features_mean_filled_array = Data_Processing_Obj.Missing_Data(Features,'mean')
Features_mean_filled_array,Labels_mean = Data_Processing_Obj.Get_Labels_and_Features()

Features_mean_filled  = np.concatenate((Features_mean_filled_array[0],Features_mean_filled_array[1],
                                        Features_mean_filled_array[2],Features_mean_filled_array[3],
                                        Features_mean_filled_array[4],Features_mean_filled_array[5],Features_mean_filled_array[6]),axis = 1)



T1_T2_Features_Mean_Fil = Features_mean_filled_array[3]
Myelin_Features_Mean_Fil = Features_mean_filled_array[0]
Curvature_Features_Mean_Fil = Features_mean_filled_array[1]
Sulc_Features_Mean_Fil = Features_mean_filled_array[2]
SC_Features_Mean_Fil = Features_mean_filled_array[4]
Vols_Features_Mean_Fil = Features_mean_filled_array[5]
Diff_Features_Mean_Fil = Features_mean_filled_array[6]


# Stacking level 2 model experiments 

Anatomical_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

T1_T2_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

Surface_Subs = [RandomForestRegressor(random_state=0),AdaBoostRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

SC_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

Vols_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

All_feats_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

Myelin_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]
Curvature_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]
Sulc_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

Diff_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]


Sub_Models = [SC_Subs,T1_T2_Subs,Myelin_Subs,Curvature_Subs,Sulc_Subs,Vols_Subs,Diff_Subs,All_feats_Subs]

Sub_Models = np.asarray(Sub_Models)


Features = [SC_Features_Mean_Fil,T1_T2_Features_Mean_Fil,Myelin_Features_Mean_Fil,Curvature_Features_Mean_Fil,
            Sulc_Features_Mean_Fil,Vols_Features_Mean_Fil,Diff_Features_Mean_Fil,Features_mean_filled]

Agg_1 = Ridge()
Agg_2 = RandomForestRegressor(random_state=0)


def train_test_feats_stack(indices_test,indices_train,Features_Array):
    
    test_features = []
    train_features = []
    
    for features in Features_Array:
        test_feats =  features[indices_test,:]
        train_feats =  features[indices_train,:]
        
        test_features.append(test_feats)
        train_features.append(train_feats)
        
        
    return train_features,test_features
        
def K_Fold_Stack(Agg_1,Agg_2,Sub_Models,Features,Labels):
    
    kf = KFold(n_splits=5)
        
    error = []

    for train_index, test_index in kf.split(Features[0]):
            
        X_train, X_test = train_test_feats_stack(test_index,train_index,Features)
        y_train, y_test = Labels[train_index], Labels[test_index]
        Stack_Obj = Stacked_Gen_2_Level(Sub_Models,Agg_1,Agg_2,'linear')
        Stack_Obj.train(X_train,y_train)
        pred = Stack_Obj.predict(X_test)
        error.append(mean_absolute_error(y_test,pred))
        print(mean_absolute_error(y_test,pred))
        
    return error,np.mean(error)

error_array,error = K_Fold_Stack(Agg_1,Agg_2,Sub_Models,Features,Labels_mean)

print("Stacking Model level 2 - Filtered Features: ", error)

kf = KFold(n_splits=5)
        
error = []

model = GradientBoostingRegressor(random_state=0)

for train_index, test_index in kf.split(Features_mean_filled):
            
    X_train, X_test = Features_mean_filled[train_index], Features_mean_filled[test_index]
    y_train, y_test = Labels_mean[train_index], Labels_mean[test_index]
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    error.append(mean_absolute_error(y_test,pred))
    print(mean_absolute_error(y_test,pred))
    
print("Single model Model: ", np.mean(error))