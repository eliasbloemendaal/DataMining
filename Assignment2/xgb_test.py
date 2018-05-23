import xgboost as xgb
import os
import pandas as pd
from random import *
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

print('hier')
dtrain = xgb.DMatrix('datasets/cv_train.txt')

"""
Parameters optimization

inspiratin: https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html

Optimize num_boost_round by setting early_stop_rounds = N, num_boost_round >> N.
Add a eval_metric (nDCG) and a test set, xgboost will automatically stop once performance stops improving for N rounds
-> For final training set num_boost_round = num_boost_round - N or num_boost_round = model.best_iteration + 1

Proposal: represent int variables as binary (directly translated)
    represent float variables using quantiles (e.g. 100 equally divided steps). Represent quantiles as binary, compute value of variable
    which belongs to the corresponding quantile
    2log(100) = +- 7 bits
    2log(1000) = +- 10 bits


Parameters:
Parameter, datatype, range, nr_bits
eta, float, [0,1], 7
gamma, float, [0,10], 7
max_depth, int, [0,15], 4
min_child_weight, int, [3, 8], 3
subsample, float, (0.5,1], 7 
colsampel, float, (0.5, 1], 7

Total length of bit string: 
35 (parameters) + 117 (? features) = 152 bits


nr_gen = 20
nr_individuals = 100

60s * 200 * 20 = 

"""

def _get_dmatrix_ranking(df, groups, weights, column_indices, target='position'):
    """
        Df containing all data, groups and weights lists, column indices boolean list
    """
    # Set target and features
    response = df[target]
    feats = df.drop(columns=target)
    #slice features
    feats = feats.iloc[:,column_indices]

    #construct dmatrix and set weights, groups
    dmatrix = xgb.DMatrix(feats.values, response.values)

    dmatrix.set_group(groups)
    dmatrix.set_weight(weights)


    return dmatrix

def get_DMatrix_data(filepath):
    df = pd.read_csv(filepath+'.csv', header = 0)
    with open(filepath+'.txt.weight', 'r') as f:
        weight = list(map(int, f.read().split()))

    with open(filepath+'.txt.group', 'r') as f:
        groups = list(map(int,f.read().split()))

    return (df, weight, groups)

def get_DMatrix_from_data(df, weight, groups, cols):

    return _get_dmatrix_ranking(df, groups, weight, cols)

def parse_predictions(df,preds):
    df['score'] = preds
    #df.sort([''])



###########################################################
# Dynamic DMatrix creation example
##########################################################
randBinList = lambda n: [randint(0,1) for b in range(1,n+1)]
# cols should be converted from binary string
cols = [bool(x) for x in randBinList(108)]

train_df, train_weight, train_groups = get_DMatrix_data('datasets/final_train')
valid_df, valid_weight, valid_groups = get_DMatrix_data('datasets/test_set')

# Every GA iteration, only cols has to be altered
dtrain = get_DMatrix_from_data(train_df, train_weight, train_groups, cols)
dvalid = get_DMatrix_from_data(valid_df, valid_weight, valid_groups, cols)

############################################################
############################################################
#Set eval list
evallist = [(dtrain, 'train'), (dvalid, 'eval')]
params = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:pairwise' ,'tree_method':'gpu_hist', 'updater':'grow_gpu','eval_metric':'ndcg@38-', 'seed':42, 'nthread':12}

num_round = 10000
bst = xgb.train(params, dtrain, num_round, evals= evallist, early_stopping_rounds = 10)
#Best NDCG score on valid set
best_score = bst.best_score
# ^ its respective prediction
pbest_pred = bst.best_iteration

preds = bst.predict(dtrain)
print(preds)
#print(len(preds))

