import xgboost as xgb

print('hier')
dtrain = xgb.DMatrix('/home/mrvoh/Documents/DataMining/Assignment2/train.txt')

#TODO: eval metric NDCG + overview params to be optimized
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:pairwise' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtrain)
print(preds)
print(len(preds))