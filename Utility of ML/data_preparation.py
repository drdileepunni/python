import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

import copy

# TODO: preprocess data for training and holdout

def dropAndChange(dataFrame, to_drop, numerics, categoricals):
    x = lambda a: pd.to_numeric(a, errors='coerce')
    y = lambda a: a.astype('category')
    changed_df = dataFrame.drop(to_drop, axis=1)
    for i in numerics:
        changed_df[i] = x(changed_df[i])
    for i in categoricals:
        changed_df[i] = y(changed_df[i])
    return changed_df  

def changeBad(dataFrame):
        dataFrame.Temperature[dataFrame[dataFrame.Temperature<95].index] = 98.4
        dataFrame.Temperature[dataFrame[dataFrame.Temperature>104].index] = 98.4
        dataFrame.SpO2[dataFrame[dataFrame.SpO2<=1].index] = (dataFrame.SpO2[dataFrame[dataFrame.SpO2<=1].index])*100
        dataFrame.SpO2[dataFrame[dataFrame.SpO2<40].index] = 98
        dataFrame.MAP[dataFrame[dataFrame.MAP<40].index] = 92
        dataFrame.MAP[dataFrame[dataFrame.MAP>300].index] = 92
        dataFrame.RR[dataFrame[dataFrame.RR>60].index] = 16
        dataFrame.FiO2[dataFrame[dataFrame.FiO2>1].index] = (dataFrame.FiO2[dataFrame[dataFrame.FiO2>1].index])/100
        dataFrame.FiO2[dataFrame[dataFrame.FiO2<0.21].index] = 0.21
        dataFrame.PaO2[dataFrame[dataFrame.PaO2>400].index] = 100
        dataFrame.PaO2[dataFrame[dataFrame.PaO2<40].index] = 100
        dataFrame.PaCO2[dataFrame[dataFrame.PaCO2>140].index] = 38
        dataFrame.PaCO2[dataFrame[dataFrame.PaCO2<15].index] = 38
        dataFrame.TLC[dataFrame[dataFrame.TLC<=50].index] = (dataFrame.TLC[dataFrame[dataFrame.TLC<=50].index])*1000
        dataFrame.TLC[dataFrame[dataFrame.TLC>200000].index] = 9000
        dataFrame.Platelets[dataFrame[dataFrame.Platelets<=10].index] = (dataFrame.Platelets[dataFrame[dataFrame.Platelets<=10].index])*100000
        dataFrame.Platelets[dataFrame[dataFrame.Platelets>2000000].index] = 220000
        dataFrame['Urine output'][dataFrame[dataFrame['Urine output']==0].index] = 1000
        dataFrame.GCS[dataFrame[dataFrame.GCS==2].index] = 3
        return dataFrame

def impute(dataFrame, num_cols, categoricals):
        fillna_freq = lambda x:x.fillna(x.value_counts().index[0])
        
        for num in num_cols:
            dataFrame[num] = dataFrame[num].fillna(dataFrame[num].median())
            
        for i in categoricals:
            dataFrame[i] = fillna_freq(dataFrame[i])
            
        return dataFrame   

def process(df):

    # Defining and naming columns
    df.columns = ['Unnamed: 0', 'Name', 'CPMRN', 'Month of Admission', 'Age', 'Gender',
        'Hospital', 'Surgery', 'Vent mode', 'GCS', 'Temperature', 'HR', 'SpO2',
        'SBP', 'MAP', 'RR', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'A-a gradient',
        'HCO3', 'Hb', 'TLC', 'Platelets', 'K', 'Na', 'Serum Cr', 'Blood Urea',
        'Bili', 'Urine output', 'Lactate', 'INR', 'Survival']
    to_drop = ['Unnamed: 0', 'Name', 'CPMRN', 'SBP', 'A-a gradient', 'Month of Admission', 
            'HCO3', 'Hospital', 'Vent mode']
    numerics = ['Age', 'Temperature', 'GCS', 'HR', 'SpO2', 'Hb', 'TLC', 'Platelets', 'K',
                'MAP', 'RR', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'Na', 'Serum Cr', 'Blood Urea', 
                'Bili', 'Urine output', 'Lactate', 'INR']
    categoricals = ['Gender', 'Surgery', 'Survival']

    # Changing types and dropping columns
    df1 = dropAndChange(df, to_drop, numerics, categoricals)

    # Removing and replacing bad values
    df2 = changeBad(df1)

    # Imputation of Na values
    df3 = impute(df2, numerics, categoricals)
    df3_unscaled = pd.DataFrame.copy(df3)

    # Encoding holdout3_unscaled
    survival = {'Alive': 0,'Expired': 1} 
    df3_unscaled['Survival'] = [survival[item] for item in df3_unscaled['Survival']] 

    # Scaling encoding and returning 
    X_h = df3_unscaled.drop('Survival', axis=1)
    y_h = df3_unscaled['Survival']
    X_h_encoded = pd.get_dummies(X_h, drop_first=True)
    cols = X_h_encoded.columns[np.arange(0,22)]
    for col in cols:
        X_h_encoded[col] = pd.DataFrame(scaler().fit_transform(pd.DataFrame(X_h_encoded[col])))
    
    return (X_h_encoded, y_h)



