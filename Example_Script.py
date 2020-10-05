"""
Created on Sun Oct  4 12:38:36 2020

@author: Shaheim Ogbomo-Harmitt

Example Script on how to use Stacking Project Code Base

"""

# IMPORTING LIBRARIES 

from Data_Processing_Ver_1_1 import *
from Stacked_Gen_2_Level import *
from Stack_K_Fold_Module import *
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# DATA PREPROCESSING

# Importing Data

T1_T2_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/T1_T2_data_v2.csv")
Myelin_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Myelin_info_LR.csv")
Curvature_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Curvature_info_LR.csv")
Sulc_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Sulc_info_LR.csv")
SC_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/SC_Full_IDs.csv")
Vols_All_Info = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Vol_data_v2.csv")

T1_T2_Full_IDs = T1_T2_All_Info.iloc[:,0].tolist()
Myelin_Full_IDs = Myelin_All_Info.iloc[:,1].tolist()
Curvature_Full_IDs = Curvature_All_Info.iloc[:,1].tolist()
Sulc_Full_IDs = Sulc_All_Info.iloc[:,1].tolist()
SC_Full_IDs = SC_All_Info.iloc[:,1].tolist()
Vols_Full_IDs = Vols_All_Info.iloc[:,0].tolist()
Diff_Full_IDs = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Diff_Full_IDs.csv").iloc[:,1].tolist()

T1_T2_Features = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/T1_T2_data_v2.csv").iloc[:,1:].values
T1_T2_Features = Fix_Data(T1_T2_Features)
Myelin_Features = pd.read_pickle("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withmyelin_rois_LR.pk1").iloc[:,:-1].values
Curvature_Features = pd.read_pickle("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withcurvature_rois_LR.pk1").iloc[:,:-1].values
Sulc_Features = pd.read_pickle("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/DHCPNDH1_DATA_LABELS_2019-06-25_1214_ga_subset_withsulc_rois_LR.pk1").iloc[:,:-1].values
SC_Features = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Processed_Data/SC_FS_Birth_Features.csv").iloc[:,1:].values
Vols_Features = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Vol_data_v2.csv").iloc[:,1:].values
Vols_Features = Fix_Data(Vols_Features)
Diff_Features = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Unprocessed Data/Diff_Data_V1.csv").iloc[:,1:].values
Diff_Features = Fix_Data(Diff_Features)

# Importing Phenotype

GA_at_Birth_raw = pd.read_csv("C:/Users/User 1/Desktop/For_Emma/Phenotypes/GA_at_Birth.csv")

GA_at_Birth_IDs = GA_at_Birth_raw.iloc[:,1].tolist()

GA_at_Birth = GA_at_Birth_raw.iloc[:,2].values


# Data_Processing module use

IDs_Array  = [Myelin_Full_IDs,Curvature_Full_IDs,Sulc_Full_IDs,T1_T2_Full_IDs,SC_Full_IDs,Vols_Full_IDs,Diff_Full_IDs]

Data_Processing_Obj = Data_Processing(IDs_Array,GA_at_Birth,GA_at_Birth_IDs)

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


# APPLYING STACKING MODEL 

T1_T2_Subs = [RandomForestRegressor(random_state=0),GradientBoostingRegressor(random_state=0)]

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

# Stacking module use

Stack_obj  = Stacked_Gen_2_Level(Sub_Models,Agg_1,Agg_2,'linear','tree')

# K-Fold module use

k_Fold_obj = K_Fold_Stack(Stack_obj,Features,Labels_mean,'Reg')

Error = k_Fold_obj.Run()
FI = k_Fold_obj.Get_FI()