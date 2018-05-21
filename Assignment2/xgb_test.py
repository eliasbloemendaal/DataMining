import xgboost as xgb
import os
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

print('hier')
dtrain = xgb.DMatrix('train.txt')

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

#TODO: eval metric NDCG + overview params to be optimized
params = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:pairwise' ,'tree_method':'gpu_hist', 'updater':'grow_gpu'}
""" Manually set eval_metric = [our ndcg function]  in params """

num_round = 2
bst = xgb.train(params, dtrain, num_round, eval = ndcg, watch_list = test_set, max_trees = 1000)

preds = bst.predict(dtrain)
print(preds)
print(len(preds))


while(True):
    print('hoi')