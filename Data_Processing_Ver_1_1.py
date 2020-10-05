"""
Created on Wed Jul 15 12:41:21 2020

@author: Shaheim Ogbomo-Harmitt

DATA PROCESSING MODULE

This module filters the IDs/sessions of data from diferent modalites, such
that it returns the subjects mutual across all modalities.
Additionally, the module can perfom mean/zero filling to all the missing
subjects across the modalities.

Inputs:
    
    Session_array - Python list of IDs for subjects for each modality
    
    Phenotype_array  - Phentoype/Labels to predict
    
    Phenotype_sessions - IDs of Phentoype/Labels to predict
    
Operations:
    
    Data_Processing(Session_array,Phenotype_array,Phenotype_sessions) -
    The constructor automatically filters the IDs/Sessions of the features
    when intialised.
                        
    Missing_Data(Features,Fill_type) - Performs feature imputation on the 
                                       feature set, zero/mean 

				       Features - Python list of all the 
						  features of each modality.

				       Fill_type - string of type of feature
						   imputation. 'mean','zero'
    
Outputs:
    
    Get_Labels - filtered Labels/Phenotypes
    
    Get_Indices - filtered indices for each modality
    
    Get_Labels_and_features - outputs mean/zero filled features for each
                              modality and the corresponding labels 
                              
"""

import numpy as np
from sklearn.impute import SimpleImputer

class Data_Processing:
    
    def __init__(self,Session_array,Phenotype_array,Phenotype_sessions):
        
        self.Session_array = Session_array
        self.Phenotype_array = Phenotype_array
        self.Phenotype_sessions = Phenotype_sessions
        self.Fill_sessions = None
        self.new_features = None
        
        Indices_Array =  []
        filtered_sessions = self.Filter_Sessions()

        for sessions in self.Session_array:
            
            Indices = self.Find_Indices(filtered_sessions,sessions)
            #print(len(Indices))
            Indices_Array.append(Indices)
            
            
        self.Final_Indices = np.transpose(np.asarray(Indices_Array))

        self.Sessions = filtered_sessions
        
    def Find_Indices(self,Sessions,Sessions_for_indices):
        
        indices = []
        
        for session in Sessions:
            
            for Index,Session_for_indices in enumerate(Sessions_for_indices):
                
                if session == Session_for_indices:
                    
                    indices.append(Index)
                    
                    break
        
        return np.asarray(indices)
        
                    
        
    def Filter_Sessions(self):
        
        Current_sessions = self.Session_array[0]
        
        for i in range(len(self.Session_array))[1:]:
            
            Next_sessions = self.Session_array[i]
            Matched_sessions = self.find_matching_sessions(Current_sessions,Next_sessions)
            Current_sessions = Matched_sessions
           
        return Current_sessions
            
            
    def find_matching_sessions(self,session_list_1,session_list_2):
    

        matching_sessions = []
    
        for i,session_1 in enumerate(session_list_1):
            
            for k,session_2 in enumerate(session_list_2):
            
                if session_1 == session_2:
    
                    matching_sessions.append(session_1)
                    
                    break 
    
        matching_sessions = np.asarray(matching_sessions).reshape(-1,1)
        
        return matching_sessions
        
    def Get_Indices(self):
        
        return self.Final_Indices
    
    def Get_IDs(self):
        
        IDs = []
        
        Sessions = self.Sessions.tolist()
        
        for Session in Sessions:
            string = Session[0]
           
            IDs.append(string)
            
        return np.asarray(IDs)
            
    def Get_Labels(self):
        
        IDs =  self.Get_IDs()
        
        indices = []
        
        for session_1 in IDs:
         
            for k,session_2 in enumerate(self.Phenotype_sessions):
                
                if session_1 == session_2:
                   
                    indices.append(k)
                
                    break 
                
        indices = np.asarray(indices)
        
        return self.Phenotype_array[indices]
    
    
    def Missing_Data(self,Features_array,fill_type):
        
        index = None 
        largest_num_sessions = 0
        
        for k,sessions in enumerate(self.Session_array):
            
            if len(sessions) > largest_num_sessions:
                
                index = k
                largest_num_sessions = len(sessions)
                
                
        session_list =  self.Session_array[index]
        self.Fill_sessions = session_list
        new_features = []
        
        if fill_type == 'mean':
            
            for k,Features in enumerate(Features_array):
                
                session_audit = self.find_data_for_sessions(session_list,self.Session_array[k])
                
                nan_filled_feats = self.fill_features_nan(session_audit,Features,self.Session_array[k],session_list)
            
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                imp.fit(nan_filled_feats)
                transformed_feats = imp.transform(nan_filled_feats)
                new_features.append(transformed_feats)
                
                
        elif fill_type == 'zeros':
            
            for k,Features in enumerate(Features_array):
                
                session_audit = self.find_data_for_sessions(session_list,self.Session_array[k])
                
                nan_filled_feats = self.fill_features_nan(session_audit,Features,self.Session_array[k],session_list)
            
                imp = SimpleImputer(missing_values=np.nan, strategy='constant')
                imp.fit(nan_filled_feats)
                transformed_feats = imp.transform(nan_filled_feats)
                new_features.append(transformed_feats)
                
        self.new_features = new_features
        
        return 0
    

    def find_data_for_sessions(self,sessions,data_sessions):
        
        session_audit = np.zeros((len(sessions)))
        
        for i,session in enumerate(sessions):
            
            for data_session in data_sessions:
                
                if session == data_session:
                    
                    session_audit[i] = 1
                    
        return session_audit
                    
    
    def fill_features_nan(self,session_audit,features,session_array,full_sessions):
        
        new_features = np.zeros((len(session_audit),features.shape[1]))
        
        for k,check in enumerate(session_audit):
            
            if check == 1:
                
                index =  np.argwhere(np.asarray(session_array) == full_sessions[k])
                new_features[k,:] = features[index[0][0],:]
                
            elif check == 0:
                
                new_features[k,:] = np.nan*features.shape[1]
                
               
        return new_features
    
    
    def Get_IDs_2(self):
        
        IDs = []
        
        Sessions = self.Fill_sessions
        
        for Session in Sessions:
            string = Session
    
            IDs.append(string)
            
        return np.asarray(IDs)
    
    def Get_Labels_and_Features(self):
        
        fil_sessions_ID =  self.Get_IDs_2()
        indices = []
        indices_2 = []
        
        for i,session_1 in enumerate(fil_sessions_ID):
        
            
            for k,session_2 in enumerate(self.Phenotype_sessions):
            
                 
                if session_1 == session_2:
                
                    indices.append(k)
                    indices_2.append(i)
                
                
        indices = np.asarray(indices)

        
        for k,features in enumerate(self.new_features):
            
            new_features = features[indices_2,:]
            self.new_features[k] = new_features
            
        
        return self.new_features,self.Phenotype_array[indices]