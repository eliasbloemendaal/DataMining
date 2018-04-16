import pandas as pd
import dataset
import datetime
import sklearn as sk
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.ensemble import AdaBoostRegressor
df = pd.read_csv('dataset_mood_smartphone (1).csv', header = 0, index_col = 0, parse_dates = ['time'])

#print(min(df.time)) # 2014-02-17
#print(max(df.time)) # 2014-06-09

#split data amongst users/participants
user_dfs = [x for _, x in df.groupby(df['id'])]
#print(len(user_dfs)) # 27 unique users


Dataset = dataset.Dataset(df)

'''for user in Dataset.users:
    #print(user.variables.head())
    user.get_sax_representation(5)
    #print(user.sax_representation.head())
    user.create_history(5, False)
    #print(user.history_response.head())
    #print(user.history_features.head())
    #print(user.variables.head())
    break'''

#model over all users
features = []
response = []
use_sax = False

for user in Dataset.users:
    user.create_history(3, use_sax)
    features.append(user.history_features)
    response.append(user.history_response)

X = pd.concat(features).fillna(value=0)
y = pd.concat(response).fillna(value=0)
if use_sax:
    X = pd.get_dummies(X)
    print(X)

X = X.as_matrix()
y = y.values

X_doc = open('X_reg.p', 'wb')
y_doc = open('y_reg.p', 'wb')
pickle.dump(X, X_doc)
pickle.dump(y, y_doc)

X_doc.close()
y_doc.close()


    
