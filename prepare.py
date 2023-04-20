import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import acquire as acq

def split_function(df, target_variable):
    train, test = train_test_split(df,
                                   random_state=123,
                                   test_size=.20,
                                   stratify= df[target_variable])
    
    train, validate = train_test_split(train,
                                   random_state=123,
                                   test_size=.25,
                                   stratify= train[target_variable])
    return train, validate, test

def prep_titanic(titanic):
    titanic = titanic.drop(columns=['embarked','class', 'age','deck'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    titanic = pd.concat([titanic, dummy_df], axis=1)
    
    return titanic

def prep_telco(telco):
    df_telco = telco.drop(columns=['internet_service_type_id' , 'contract_type_id', 'payment_type_id'])

    df_telco['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    df_telco['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    df_telco['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    df_telco['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    df_telco['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    df_telco['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(df[['phone_service' , 
                                  'tech_support' , 
                                  'multiple_lines' , 
                                  'online_security' , 
                                  'online_backup' , 
                                  'device_protection' , 
                                  'streaming_tv' , 
                                  'streaming_movies' , 
                                  'churn']], 
                              drop_first=[True])
    
    df_telco = pd.concat( [df_telco, dummy_df], axis=1)
    
    df_telco.total_charges = df_telco.total_charges.str.replace(' ', '0').astype(float)
    
    return telco

def prep_iris(df_iris):
    '''
    This function will drop any duplicate observations, 
    drop ['deck', 'embarked', 'class', 'age'], fill missing embark_town with 'Southampton'
    and create dummy vars from sex and embark_town.'''
    df_iris = df_iris.drop_duplicates()
    df_iris = df_iris.drop(columns=['species_id'])
    dummy_df = pd.get_dummies(df_iris[['species_name']], drop_first=True)
    df_iris = pd.concat([df_iris, dummy_df], axis=1)
    
    return df_iris