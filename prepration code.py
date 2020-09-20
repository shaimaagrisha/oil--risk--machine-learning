import numpy as np
import pandas as pd


df = pd.read_csv('hcc.csv')

types = df.dtypes

df = df.iloc[:,1:51]

def cols_exctractor(data):
    cols = list()
    for col in data.columns:
        cols.append(col)
    return cols

columns_names = cols_exctractor(df)

def cat_sep(data , columns):
    count = data[columns].nunique()
    cat = list()
    num = list()
    for col in count.index.values:
        if count.loc[col] > 6 :
            num.append(col)
        else :
            cat.append(col)
    cat.remove('Class')
    return num , cat

num , cat  = cat_sep(df ,columns_names )

def missing_replace (data) :
    data = data.copy()
    missing = df.isna().sum()
    for col in missing.index.values:
        if col in cat and missing.loc[col] > 0 :
            data[col] = data[col].fillna(method = 'ffill')
            data[col] = data[col].fillna(method = 'bfill')
        elif col in num and missing.loc[col] > 0 :
            if data[col].mean() - data[col].median() >= 100 :
                data[col] = data[col].fillna(data[col].median())
            else :
                data[col] = data[col].fillna(data[col].mean())
    return data
    
df = missing_replace(df)

from scipy.stats import zscore

def outlier_drop(data , columns):
    data = data.copy()
    for i in columns :
        data['z'] = zscore(data[i])
        data['z'] = data['z'].apply(lambda x : x >=3 or x <=-3)
        data = data[data['z'] == False]
        data = data.drop('z' , axis =1)
    return data
df = outlier_drop(df , num)

def type_conv (data , cat_cols , num_cols):
    data = data.copy()
    data[cat_cols] = data[cat_cols].astype('str')
    data[num_cols] = data[num_cols].astype('float')
    return data
df = type_conv(df , cat , num)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


def scaler_dummy_label(data , cat_cols , num_cols):
    data = data.copy()
    scaler = MinMaxScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    encoder = LabelEncoder()
    for col in cat_cols :
        data[col] = encoder.fit_transform(data[col])
    count = data[cat_cols].nunique()
    data[cat_cols] = data[cat_cols].astype('category')
    dummy_cols = list()
    for col in count.index.values:
        if count.loc[col] > 2 :
            dummy_cols.append(col)
    dummies = data[dummy_cols]
    dummies_df = pd.get_dummies(dummies)
    data = data.drop(dummy_cols , axis =1)
    data = data.join(dummies_df )
    return data 

df = scaler_dummy_label(df , cat ,num)

from sklearn.model_selection import train_test_split

def splitter(data , test_ratio , output):
    x = data.drop(output , axis=1)
    y = data[output]
    x_train,x_test , y_train,y_test = train_test_split(x,y,test_size = test_ratio)
    return x_train,x_test , y_train,y_test
x_train,x_test , y_train,y_test = splitter(df , .2 ,'Class')



