#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:43:26 2020

@author: Ogbomo-Harmitt
"""

## DATA IMPORTING 

import pandas as pd
import numpy as np
from Data_Processing_Ver_1_1 import *
from Single_Model_Exp_Module import *
from vis_module import *
from Stacked_Gen_2_Level import *
from Stacked_Gen import *
from Vis_Func_mod import *
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


# DATA PREPROCESSING 

Right_Labels = pd.read_csv("Surface_Right_Labels.csv").iloc[:,1].tolist()
Left_Labels = pd.read_csv("Surface_Left_Labels.csv").iloc[:,1].tolist()

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
Myelin_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withmyelin_rois_LR.pk1").iloc[:,:-1].values
Curvature_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withcurvature_rois_LR.pk1").iloc[:,:-1].values
Sulc_Features = pd.read_pickle("DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withsulc_rois_LR.pk1").iloc[:,:-1].values
SC_Features = pd.read_csv("SC_FS_Birth_Features.csv").iloc[:,1:].values
Vols_Features = pd.read_csv("Vol_data_v2.csv").iloc[:,1:].values
Diff_Features = pd.read_csv("Diff_Data_V1.csv").iloc[:,1:].values

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
        
    FI = []

    for train_index, test_index in kf.split(Features[0]):
            
        X_train, X_test = train_test_feats_stack(test_index,train_index,Features)
        y_train, y_test = Labels[train_index], Labels[test_index]
        Stack_Obj = Stacked_Gen_2_Level(Sub_Models,Agg_1,Agg_2,'linear')
        Stack_Obj.train(X_train,y_train)
        Current_FI = Stack_Obj.Get_Feature_Importances()
        FI.append(Current_FI)
        
    return FI

FI = K_Fold_Stack(Agg_1,Agg_2,Sub_Models,Features,Labels_mean)


def Add_Importances(Array_1,Array_2):
    
    new_FI = []
    
    for mod_index,mod_1 in enumerate(Array_1):
        
        mod_2 = Array_2[mod_index]
        
        new_FI.append(np.add(mod_1,mod_2))
        
    return new_FI
        

def Get_Mean_FI(Importances_Array):
    
    FI_0 = Importances_Array[0]
    
    array = [1,2,3,4]
    
    for i in array:
        
        current_FI = FI[i]
        
        FI_0 = Add_Importances(FI_0,current_FI)
        
    return FI_0

test = Get_Mean_FI(FI)

vis_obj = Vis_Mod(test)
Ind_imp,AF_imp = vis_obj.Data_Preprocessing()


###### Visualisation 

# All Feats Sub Model

Mod = AF_imp[0][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'L_MYELIN_AF_Birth.func.gii'
labels  = '40.L.voronoi.label.gii'
num_channel = 1


Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()

Mod = AF_imp[1][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'R_MYELIN_AF_Birth.func.gii'
labels  = '40.R.voronoi.label.gii'
num_channel = 1

Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()

Mod = AF_imp[2][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'L_CURV_AF_Birth.func.gii'
labels  = '40.L.voronoi.label.gii'
num_channel = 1

Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()

Mod = AF_imp[3][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'R_CURV_AF_Birth.func.gii'
labels  = '40.R.voronoi.label.gii'
num_channel = 1

Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()


Mod = AF_imp[4][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'L_SULC_AF_Birth.func.gii'
labels  = '40.L.voronoi.label.gii'
num_channel = 1

Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()

Mod = AF_imp[5][0]
Mod = np.asarray(Mod)

features = np.argsort(Mod)[::-1]
importances = Mod[np.argsort(Mod)[::-1]]
feature_names = Right_Labels
Outname = 'R_SULC_AF_Birth.func.gii'
labels  = '40.R.voronoi.label.gii'
num_channel = 1

Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Surface()



T1_T2 = Ind_imp[8][0]
T1_T2 = np.asarray(T1_T2)

features = np.argsort(T1_T2)[::-1]
importances = T1_T2[np.argsort(T1_T2)[::-1]]
feature_names = np.arange(87)
Outname = 'Diff_IND.nii.gz'
labels  = "T1_T2_test.nii.gz"

num_channel = 1


Vis_Full_Obj = Vis_Func_Mod(Outname,features,feature_names,labels,importances,num_channel)
Vis_Full_Obj.Run_Ana()