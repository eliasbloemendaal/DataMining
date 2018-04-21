import pandas as pd
import dataset
import datetime
import sklearn as sk
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from basic_ml import do_ml

from sklearn.ensemble import AdaBoostRegressor
df = pd.read_csv('OutliersData.csv', header = 0, index_col = 0, parse_dates = ['time'])
df = df.drop(columns=['X'])

#Filter time outliers
df = df[df.time > '01-01-2013']
#Drop outliers
df = df[df.outlier == 0]
df.drop(columns=['outlier'])


#print(min(df.time)) # 2014-02-17
#print(max(df.time)) # 2014-06-09

#split data amongst users/participants
user_dfs = [x for _, x in df.groupby(df['id'])]
#print(len(user_dfs)) # 27 unique users


Dataset = dataset.Dataset(df)

#for user in Dataset.users:
#    user.variables['id'] = user.id

#pd.concat([user.variables for user in Dataset.users]).to_csv('RuthsList.csv')

'''
Utilize engineered features:
    read in engineered df from csv
    set df.Users = [x for _, x in df.groupby(df['id'])]
    Create history for days in range(5)
    Perform basic_ml --> run...
    Aggregate results

'''
#import extra day features
extra_features = pd.read_csv('assignment1_ruth/extra_data.csv', header = 0, index_col = 0, parse_dates = ['datetime'], usecols =[x for x in range(1,11)])

#model over all users
use_sax = False

#join extra features
for user in Dataset.users:
    user.variables = user.variables.join(extra_features)

for day in [2,3,4,5,6]:
    features = []
    response = []
    for user in Dataset.users:
        #user.variables = user.variables.join(extra_features)
        user.create_history(day, use_sax)
        features.append(user.history_features)
        response.append(user.history_response)

    X = pd.concat(features).fillna(value=0)
    y = pd.concat(response).fillna(value=0)

    X.to_csv('RuthsList-extended.csv')
    if use_sax:
        X = pd.get_dummies(X)

    X = X.as_matrix()
    y = y.values

    X_doc = open('X_reg.p', 'wb')
    y_doc = open('y_reg.p', 'wb')
    pickle.dump(X, X_doc)
    pickle.dump(y, y_doc)

    X_doc.close()
    y_doc.close()
    do_ml(day)


    
