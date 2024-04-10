import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.preprocessing import *
from sklearn.metrics import *
from scipy.stats import * 
from sklearn.metrics import *
import joblib
import pickle
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


def One_hot_encoder(train, *args, variable):
    train_copy = train.copy()
    new_args = [df.copy() for df in args]

    for feature in variable:
        unique = sorted(train[feature].unique().tolist())

        for i in unique:
            train_copy[feature + '_' + str(i)] = np.where(train[feature] == i, 1, 0).astype(float)

        if args:
            for df, df_copy in zip(args, new_args):
                for i in unique:
                    df_copy[feature + '_' + str(i)] = np.where(df[feature] == i, 1, 0).astype(float)

    train_copy = train_copy.drop(variable, axis = 1)
    new_args = [df.drop(variable, axis = 1) for df in new_args]

    if args:
        return [train_copy, *new_args]
    else:
        return train_copy



  

def Scaling(train, *args, method = 'scaling', binary = None, exception = None, scaler = StandardScaler(), save = False):

    train = train.reset_index(drop = True)
    
    original_variable = train.columns

    if exception != None:
        train_exception = train[exception]
        train = train.drop(exception, axis = 1)

    if binary == 'all':
        if method == 'scaling':
            train_total = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.transform(train), columns=train.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_total = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns)   
    
    else:    
        if binary is None:
            binary = list()
            for i in train.columns:
                if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                    binary.append(i)
        else:
            binary = binary
        
        train_binary = train[binary].reset_index(drop=True)
        train_conti = train.drop(binary, axis=1).reset_index(drop=True)
        
        if method == 'scaling':
            train_conti = pd.DataFrame(scaler.fit_transform(train_conti), columns=train_conti.columns)
        elif method == 'apply':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.transform(train_conti), columns=train_conti.columns)
        elif method == 'inverse':
            scaler = joblib.load(scaler + '.joblib')
            train_conti = pd.DataFrame(scaler.inverse_transform(train_conti), columns=train_conti.columns)    
        
        train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)
        
    if exception != None:
        train_total[exception] = train_exception
        
    train_total = train_total[original_variable]
                
    scaled_args = list()    
    if args:
        for data in args:
            data = data.reset_index(drop = True)
            if exception != None:
                data_exception = data[exception]
                data = data.drop(exception, axis = 1)

            if binary == 'all':
                data_total = pd.DataFrame(scaler.transform(data), columns=train.columns)        

            else:            
                data_binary = data[binary].reset_index(drop=True)
                data_conti = data.drop(binary, axis=1).reset_index(drop=True)

                if method == 'scaling' or method == 'apply':
                    data_conti = pd.DataFrame(scaler.transform(data_conti), columns=train_conti.columns)        
                elif method == 'inverse':
                    data_conti = pd.DataFrame(scaler.inverse_transform(data_conti), columns=train_conti.columns)   
                
                data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)
                
            if exception != None:
                data_total[exception] = data_exception
            
            data_total = data_total[original_variable]
            
            scaled_args.append(data_total)
        
    if save != False:
        joblib.dump(scaler, save + '.joblib')
    
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total
    
    
    
def Drop_only_one_columns(data):

    only_one_value = []
    for i in data.columns:
        if len(data[i].unique()) == 1:
            only_one_value.append(i)
    
    data = data.drop(only_one_value, axis = 1)
    
    after = len(data.columns)
            
    print("Removed Columns :", only_one_value)
    print("Total Columns N :", after)
    
    print('---------------------------------------------')
    
    return data


def Log_transformation(train, *args, binary = None):
    original_variable = train.columns
    
    if binary is None:
        binary = list()
        for i in train.columns:
            if ((train[i].min() == 0) & (train[i].max() == 1)) or (len(train[i].unique()) == 1):
                binary.append(i)
    else:
        binary = binary
            
    train_binary = train[binary].reset_index(drop=True)
    train_conti = train.drop(binary, axis=1).reset_index(drop=True)
    
    train_conti = pd.DataFrame(np.log(train_conti), columns=train_conti.columns)
    train_total = pd.concat([train_conti, train_binary], axis=1).astype(float)[original_variable]
    train_total = pd.DataFrame(np.where(np.isinf(train_total), 0, train_total), columns = original_variable)
    train_total = train_total.fillna(0)
            
    scaled_args = list()    
    if args:
        for data in args:
            data_binary = data[binary].reset_index(drop=True)
            data_conti = data.drop(binary, axis=1).reset_index(drop=True)
        
            data_conti = pd.DataFrame(np.log(data_conti), columns=train_conti.columns)        
            data_total = pd.concat([data_conti, data_binary], axis=1).astype(float)[original_variable]
            data_total = pd.DataFrame(np.where(np.isinf(data_total), 0, data_total), columns = original_variable)
            data_total = data_total.fillna(0)
        
            scaled_args.append(data_total)
            
    if args:
        return [train_total, *scaled_args]
    else:
        return train_total
    
    
    
def find_continuous_col(data):
    continuous_col = []
    for i in data.columns:
        if len(data[i].unique()) > 2:
            continuous_col.append(i)
       
    return continuous_col


def ANOVA_F(df, continuous_col):
    
    X_train, y_train = df[continuous_col].drop('Income', axis = 1), df['Income']

    fs = SelectKBest(score_func=f_regression, k='all')
    fs.fit(X_train, y_train)

    cols_idxs = fs.get_support(indices=True)
    features_name = X_train.columns[cols_idxs]
    scores = fs.scores_

    results = pd.DataFrame({'Feature': features_name, 'Score': scores})
    
    results = results.sort_values(by = 'Score', ascending=False).reset_index(drop = True)
    
    results = results[results['Score'] != 0]

    return results


def Mutual_info(df, continuous_col):
        
    X_train, y_train = df.drop('Income', axis = 1), df['Income']
    
    cat_order = []
    for i, col in enumerate(X_train.columns):
        if col not in continuous_col:
            cat_order.append(i)
       
    mutual_info = mutual_info_regression(X_train, y_train, discrete_features=cat_order)

    mutual_info_df = pd.DataFrame(mutual_info, columns=['Score'], index=X_train.columns)

    mutual_info_df.reset_index(inplace=True)
    
    mutual_info_df.columns = ['Feature', 'Score']
    
    mutual_info_df = mutual_info_df.sort_values(by = 'Score', ascending=False)
    
    mutual_info_df = mutual_info_df[mutual_info_df['Score'] != 0]

    return mutual_info_df



def Feature_selection(train, top_n):

    continuous_col = find_continuous_col(train)

    anova = ANOVA_F(train, continuous_col)
    mi = Mutual_info(train, continuous_col)    
    
    anova_scaled = Scaling(anova, scaler = MinMaxScaler(), exception='Feature')
    mi_scaled = Scaling(mi, scaler = MinMaxScaler(), exception='Feature')

    df = pd.concat([anova_scaled, mi_scaled])

    result = df.groupby('Feature')['Score'].sum().reset_index()

    feature = result.sort_values(['Score'], ascending = False).head(top_n)['Feature'].tolist()

    return feature


