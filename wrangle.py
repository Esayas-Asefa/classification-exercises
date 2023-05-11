import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

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
    titanic = titanic.drop(columns=['passenger_id', 'sibsp', 'parch', 'embarked', 'class', 'deck', 'alone'])
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

def split_my_data(df):
    X = df.drop(['churn_Yes', 'tenure', 'monthly_charges', 'total_charges',
             'phone_service_Yes', 'multiple_lines_No phone service', 
            'multiple_lines_Yes', 'online_security_No internet service',
       'online_security_Yes', 'online_backup_No internet service',
       'online_backup_Yes', 'device_protection_No internet service',
       'device_protection_Yes', 'tech_support_No internet service',
       'tech_support_Yes', 'streaming_tv_No internet service',
       'streaming_tv_Yes', 'streaming_movies_No internet service',
       'streaming_movies_Yes', 'paperless_billing_Yes',
       'contract_type_One year', 'contract_type_Two year',
       'internet_service_type_Fiber optic', 'internet_service_type_None',
       'payment_type_Credit card (automatic)', 'payment_type_Electronic check',
       'payment_type_Mailed check'], axis=1)
    Y = df.churn_Yes
    
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    return X, Y

def plot_gender_churn(X_train, y_train):
    sns.barplot(x = X_train.gender_Male , y = y_train.churn_Yes)
    # Add labels and title
    plt.title('Barplot of Gender and Churn')
    # Show the plot
    plt.show()
    
def plot_senior_citizen_Yes_churn(X_train, y_train):
    sns.barplot(x = X_train.senior_citizen_Yes , y = y_train.churn_Yes)
    # Add labels and title
    plt.title('Barplot of those are Senior Citizens and Churn')
    # Show the plot
    plt.show()
    
def plot_partner_Yes_churn(X_train, y_train):
    sns.barplot(x = X_train.partner_Yes , y = y_train.churn_Yes)
    # Add labels and title
    plt.title('Barplot of those who have Partners and Churn')
    # Show the plot
    plt.show()
    
def plot_dependents_Yes_churn(X_train, y_train):
    sns.barplot(x = X_train.dependents_Yes , y = y_train.churn_Yes)
    # Add labels and title
    plt.title('Barplot of those who have Dependents and Churn')
    # Show the plot
    plt.show()

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

def train_validate_accuracy_dt(X_train, y_train, X_validate, y_validate):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree.score(X_train, y_train)
    tree.score(X_validate, y_validate)
    #baseline_accuracy = (df.churn_Yes == 0).mean().round(5)
    for x in range(1,4):
    #     print(x)
        tree = DecisionTreeClassifier(max_depth=x)
        tree.fit(X_train, y_train)
        acc = tree.score(X_train, y_train)
        val = tree.score(X_validate, y_validate)
        print(f'for depth of {x:4}, the train accuracy is {round(acc,5)}')
        print(f'for depth of {x:4}, the validate accuracy is {round(val,5)}')
        #print(f'baseline_accuracy {baseline_accuracy}')
        print()
def train_validate_accuracy_rf(X_train, y_train, X_validate, y_validate):
    tree = RandomForestClassifier()
    tree.fit(X_train, y_train)
    tree.score(X_train, y_train)
    tree.score(X_validate, y_validate)     
    #baseline_accuracy = (df.churn_Yes == 0).mean().round(5)
    for x in range(1,5):
#         print(x)
        tree = RandomForestClassifier(max_depth=x)
        tree.fit(X_train, y_train)
        acc = tree.score(X_train, y_train)
        val = tree.score(X_validate, y_validate)
        print(f'for depth of {x:2}, the train accuracy is {round(acc,5)}')
        print(f'for depth of {x:2}, the validate accuracy is {round(val,5)}')
        #print(f'baseline_accuracy {baseline_accuracy}')
        print()
        
def split_my_data_m(df):
    X_m = df.drop(['churn_Yes', 'tenure', 'monthly_charges', 'total_charges',
             'phone_service_Yes', 'multiple_lines_No phone service', 
            'multiple_lines_Yes', 'online_security_No internet service',
       'online_security_Yes', 'online_backup_No internet service',
       'online_backup_Yes', 'device_protection_No internet service',
       'device_protection_Yes', 'tech_support_No internet service',
       'tech_support_Yes', 'streaming_tv_No internet service',
       'streaming_tv_Yes', 'streaming_movies_No internet service',
       'streaming_movies_Yes', 'paperless_billing_Yes',
       'contract_type_One year', 'contract_type_Two year',
       'internet_service_type_Fiber optic', 'internet_service_type_None',
       'payment_type_Credit card (automatic)',
       'payment_type_Mailed check'], axis=1)
    Y_m = df.churn_Yes
    
    X_m = pd.DataFrame(X_m)
    Y_m = pd.DataFrame(Y_m)
    return X_m, Y_m