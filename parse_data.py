import pandas as pd
import dataset
import datetime

df = pd.read_csv('dataset_mood_smartphone (1).csv', header = 0, index_col = 0, parse_dates = ['time'])

#print(min(df.time)) # 2014-02-17
#print(max(df.time)) # 2014-06-09

#split data amongst users/participants
user_dfs = [x for _, x in df.groupby(df['id'])]
print(len(user_dfs)) # 27 unique users

#for user in user_dfs:
#    print((min(user.time), max(user.time))) # variation in length of experiment per user

#print(df.head(10))

Dataset = dataset.Dataset(df)

for user in Dataset.users:
    #print(user.variables.head())
    user.get_sax_representation(5)
    #print(user.sax_representation.head())
    user.create_history(5, False)
    #print(user.history_response.head())
    #print(user.history_features.head())
    #print(user.variables.head())
    break