from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset

import argparse
import os
import joblib
import pandas as pd
import numpy as np
from azureml.core import Workspace, Dataset
# load dataset

def year_count(variable,df):
    df[variable] = df[variable].astype("int")
    df[variable] = 2019 - df[variable]
    return df[variable]

def geo_zones(state):
    North_Central = ['KOGI','BENUE','KWARA','NASSARAWA','NIGER','PLATEAU','FCT']
    North_East = ['ADAMAWA','BAUCHI','BORNO','GOMBE','TARABA','YOBE']
    North_West = ['JIGAWA','KADUNA','KANO','KATSINA','KEBBI','SOKOTO','ZAMFARA']
    South_East = ['ABIA','ANAMBRA','EBONYI','ENUGU','IMO']
    South_South = ['AKWA IBOM','BAYELSA','CROSS RIVER','RIVERS','DELTA','EDO']
    South_West = ['EKITI','LAGOS','OGUN','ONDO','OSUN','OYO']
    
    if state in North_Central:
        return "North_Central"
    elif state in North_East :
        return "North_East"
    elif state in North_West:
        return "North_West"
    elif state in South_East:
        return "South_East"
    elif state in South_South:
        return "South_South"
    else:
        return "South_West"

def clean_data(data):

    data = data.to_pandas_dataframe()
    x_df = data
    x_df['Qualification'].fillna(x_df['Qualification'].value_counts().index[0],inplace=True)
    degree_holder ={"MSc, MBA and PhD":"Yes","Non-University Education":"No","First Degree or HND":"Yes"}
    x_df['Qualification'] = x_df['Qualification'].map(degree_holder)
    x_df['age_recruitment'] = year_count('Year_of_recruitment',x_df)
    x_df.drop(labels = ['Year_of_recruitment','Year_of_birth'], axis = 1, inplace = True)
    x_df['State_Of_Origin'] = x_df['State_Of_Origin'].apply(lambda x: geo_zones(x))
    x_df.drop(['EmployeeNo'],axis=1,inplace=True)
    column = ['Gender','Division','State_Of_Origin','Qualification','Channel_of_Recruitment','Last_performance_score','Foreign_schooled','Marital_Status','Past_Disciplinary_Action','Previous_IntraDepartmental_Movement','No_of_previous_employers']
    x_df=pd.get_dummies(x_df, columns=column,drop_first=True)
    yt=x_df.pop("Promoted_or_Not")

    return x_df, yt


url = "https://raw.githubusercontent.com/Opiano1/nd00333-capstone/master/starter_file/data/train.csv"
data = TabularDatasetFactory.from_delimited_files(url)

x, y = clean_data(data)

# split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y)

run = Run.get_context() 

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))


    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #Save model for current iteration using C and max_iter values
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.C,args.max_iter))

if __name__ == '__main__':
    main()

