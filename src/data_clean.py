from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def data_cleaner(df):
    
    #Convert last_trip_date and signup_date to datetime object
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    
    
    #Convert last_trip_date column to int
    df['last_trip_date'] = df['last_trip_date'] <= '2014-06-01'
    df['last_trip_date'] = df['last_trip_date'].astype(int)
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)
    
    
    #Drop Columns
    df.drop(columns=['signup_date', 'avg_surge', 'phone'], inplace=True)
    
    #Rename target values
    df.rename(columns={'last_trip_date': 'target'}, inplace=True)
    
    #Filter out outliers
    df = df[df['surge_pct'] < 50]
    
    #Hot encode categorical features
    df = pd.get_dummies(df, columns=['city'])

    # Fill in missing values
    imputer = SimpleImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    #Remove Duplicate Rows
    df.drop_duplicates()
    
    
    
    return df
    

