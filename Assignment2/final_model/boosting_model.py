import xgboost as xgb
import math
import pandas as pd

class BoostingModel:

	def _get_dmatrix_ranking(self, df, groups, weights, column_indices, target='position'):
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


	def get_DMatrix_data(self, filepath):
	    df = pd.read_csv(filepath+'.csv', header = 0)
	    with open(filepath+'.txt.weight', 'r') as f:
	        weight = list(map(int, f.read().split()))

	    with open(filepath+'.txt.group', 'r') as f:
	        groups = list(map(int,f.read().split()))

	    return (df, weight, groups)


	def get_DMatrix_from_data(self, df, weight, groups, cols):
	    return _get_dmatrix_ranking(df, groups, weight, cols)


	def get_features(self, genotype):
		return [bool(i) for i in genotype[:117]]


	def get_params(self, genotype):
		eta = int(''.join([str(sub) for sub in genotype[117:124]]),2) / (2**7-1)
		gamma = int(''.join([str(sub) for sub in genotype[124:131]]),2) / (2**7-1) * 10
		max_depth = int(''.join([str(sub) for sub in genotype[131:135]]), 2)
		min_child_weight = int(''.join([str(sub) for sub in genotype[135:138]]), 2) + 1
		subsample = int(''.join([str(sub) for sub in genotype[138:145]]), 2) / (2**7 - 1) / 2 + 0.5
		colsample = int(''.join([str(sub) for sub in genotype[145:152]]), 2) / (2**7 - 1) / 2 + 0.5

		return eta, gamma, max_depth, min_child_weight, subsample, colsample

	def run_on_genotype(self, genotype):
		eta, gamma, max_depth, min_child_weight, subsample, colsample = self.get_params(genotype)
		features = self.get_features(genotype)
		
		dtrain = self.get_DMatrix_from_data(self.df, self.weight, self.groups, features)


		#TODO: eval metric NDCG + overview params to be optimized
		param = {'max_depth':max_depth,
				 'eta':eta,
				 'silent':1,
				 'gamma':gamma,
				 'objective':'rank:pairwise',
				 'subsample':subsample,
				 'colsample':colsample}

		num_round = 10000
		bst = xgb.train(param, dtrain, num_round)


		preds = bst.predict(dtrain)
		print(preds)
		print(len(preds))
		ndcg = bst.best_score

		####

		dtrain = get_DMatrix_from_data(self.train_df, self.train_weight, self.train_groups, cols)
		dvalid = get_DMatrix_from_data(self.valid_df, self.valid_weight, self.valid_groups, cols)


		evallist = [(dtrain, 'train'), (dvalid, 'eval')]
		params = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'rank:pairwise' ,'tree_method':'gpu_hist', 'updater':'grow_gpu','eval_metric':'ndcg@38-', 'seed':42, 'nthread':12}

		num_round = 10000
		bst = xgb.train(params, dtrain, num_round, evals= evallist, early_stopping_rounds = 10)
		#Best NDCG score on valid set
		best_score = bst.best_score
		# ^ its respective prediction
		pbest_pred = bst.best_iteration

		return ndcg





	def __init__(self):
		file_train = '/home/mrvoh/Documents/DataMining/Assignment2/'
		file_valid = '/home/mrvoh/Documents/DataMining/Assignment2/'
		self.train_df, self.train_weight, self.train_groups = self.get_DMatrix_data(file_train)
		self.valid_df, self.valid_weight, self.valid_groups = self.get_DMatrix_data(file_valid)
		

