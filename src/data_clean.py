from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer



def data_cleaner(df):
    imputer = SimpleImputer()
    
    #Convert last_trip_date and signup_date to datetime object
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    
    
    #Convert last_trip_date column to int
    df['last_trip_date'] = df['last_trip_date'] >= '2014-06-01'
    df['last_trip_date'] = df['last_trip_date'].astype(int)
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)
    
    
    #Drop Columns
    df.drop(columns=['signup_date'], inplace=True)
    
    return df
    

