import pandas as pd
import os
from sklearn.model_selection import train_test_split

from termcolor import colored


import env
import numpy as np


def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

def get_titanic_data():
    url = env.get_connection_url('titanic_db')
    filename = 'titanic.csv'
    query = 'select * from passengers'

    df = check_file_exists(filename, query, url)

    return df


def get_iris_data():
    url = env.get_connection_url('iris_db')
    query = '''
            select * from measurements
                join species
                    using (species_id)
            '''
    filename = 'iris.csv'
    df = check_file_exists(filename, query, url)
    return df



def get_telco_churn():
    url = env.get_connection_url('telco_churn')
    query = ''' select * from customers
	join contract_types
		using (contract_type_id)
	join internet_service_types
		using (internet_service_type_id)
	join payment_types
		using (payment_type_id)
        '''
    filename = 'telco.csv'
    df = check_file_exists(filename, query, url)

    return df

def prep_iris():
    iris = get_iris_data()
    iris = iris.drop(columns=['species_id', 'measurement_id'])
    iris = iris.rename(columns={'species_name': 'species'})
    num=iris.select_dtypes(include="number")
    char=iris.select_dtypes(include="object")
    char_1 = pd.get_dummies(char)
    char = pd.concat([char, char_1], axis=1)
    iris_clean = pd.concat([num, char], axis=1)
    return iris_clean



def prep_telco():
    telco = get_telco_churn()
    telco = telco.drop(columns=['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    telco['senior_citizen'] = telco['senior_citizen'].replace({0: 'No', 1: 'Yes'})
    telco['total_charges'] = telco['total_charges'].replace({' ': 0})
    telco['total_charges'] = telco['total_charges'].astype('float')
    num=telco.select_dtypes(include="number")
    char=telco.select_dtypes(include="object")
    char_1 = pd.get_dummies(char, drop_first=True)
    char = pd.concat([char, char_1], axis=1)
    telco_clean = pd.concat([num, char_1], axis=1)
    return telco_clean

def prep_titanic(titanic):
    titanic = titanic.drop(columns=['passenger_id', 'sibsp', 'sex', 'parch', 'embarked', 'class', 'deck', 'embark_town', 'alone'])
    dummy_df = pd.get_dummies(data=titanic[['sex','embark_town']], drop_first=True)
    titanic = pd.concat([titanic], axis=1)
    
    return titanic

def test_train(df, target):

    temp_train, test = train_test_split(df,  test_size= .2, random_state=123, stratify=df[target])
    train, validate = train_test_split(temp_train, test_size= .25, random_state=123, stratify=temp_train[target])
    return train, validate, test

def eval_results(p):
    alpha = .05
    if p < alpha:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")
def prep_data(df):
    sum_null = df.isnull().sum()
    df = df.drop(columns=sum_null[sum_null > 0].index)
    print(colored(f'Columns with null values: {sum_null[sum_null > 0].index}', 'red'))
    return df

def check_nulls(df):
    return df.isnull().sum()

def check_duplicates(df):
    if df.nunique().sum() == df.shape[0]:
        print('No duplicates')
    else:
        print('Duplicates exist')

def train_validate_test_split(X_all, Y):
    '''
    This function takes in a dataframe, the target variable, and a seed for reproducibility.
    It will split the data into train, validate, and test datasets.
    '''

    X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.2, random_state=123, stratify=Y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)
    return X_train, X_validate, X_test, y_train, y_validate, y_test