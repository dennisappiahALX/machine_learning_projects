import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from src import config
import joblib

def feature_process(data, mode):
    df = pd.read_csv(config.TRAIN_DATA)
    # Converting Each column to appropriate datatypes
    date_cols = ['GarageYrBlt', 'YrSold','YearBuilt', 'YearRemodAdd']
    cat_cols = [col for col in df.columns if df.dtypes[col]== 'object']
    num_cols = [col for col in df.columns if col not in date_cols + cat_cols + ['Id']]

    for col in df.columns:
      if col in date_cols:
        df[col] = pd.to_datetime(df[col])
      if col in cat_cols:
        df[col] = df[col].astype('object')

    # MIssing Values Treatments
    # For cat cols and date cols fill in with mode
    for col in df.columns:
        if col in date_cols + cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        if col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    #dropping some columns
    df.drop(date_cols, axis=1, inplace=True)
    df.drop(columns=['Id'], inplace=True)

    ##Converting categorical columns with OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    if mode == 'train':
        df[cat_cols] = ordinal_encoder.fit_transform(df[cat_cols])
    else:
        ordinal_encoder = joblib.load(config.MODELS_PATH + 'feature_encoders.pkl')
        df[cat_cols] = ordinal_encoder.transform(df[cat_cols])

    df.to_csv(config.TRAIN_PROCESSED_DATA, index= False)
    if mode == 'train':
        joblib.dump(ordinal_encoder, config.MODELS_PATH + 'feature_encoders.pkl')


feature_process(config.TRAIN_DATA, 'train')