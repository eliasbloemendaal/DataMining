import xgboost as xgb
import math
import pandas as pd
import time
import gc
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
		return self._get_dmatrix_ranking(df, groups, weight, cols)


	def get_features(self, genotype):
		return [bool(i) for i in genotype[:117]]


	def get_params(self, genotype):
		eta = int(''.join([str(sub) for sub in genotype[117:124]]),2) / (2**7-1)
		gamma = int(''.join([str(sub) for sub in genotype[124:131]]),2) / (2**7-1) * 10
		max_depth = int(''.join([str(sub) for sub in genotype[131:135]]), 2) +1 
		min_child_weight = int(''.join([str(sub) for sub in genotype[135:138]]), 2) + 1
		subsample = int(''.join([str(sub) for sub in genotype[138:145]]), 2) / (2**7 - 1) / 2 + 0.5
		colsample = int(''.join([str(sub) for sub in genotype[145:152]]), 2) / (2**7 - 1) / 2 + 0.5

		return eta, gamma, max_depth, min_child_weight, subsample, colsample

	def predict(self, model, dmatrix, df, weights):

		df['pred_score'] = model.predict(dmatrix) #predict returns list of scores per row

		result = df[['srch_id','prop_id', 'pred_score']]
		result['weight'] = weights

		
		result = result.sort_values(by=['srch_id', 'pred_score'], ascending=False, kind='heapsort')

		return result




	def run_on_genotype(self, genotype):
		eta, gamma, max_depth, min_child_weight, subsample, colsample = self.get_params(genotype)
		features = self.get_features(genotype)
		
		#dtrain = self.get_DMatrix_from_data(self.df, self.weight, self.groups, features)
		print('nr features: {}'.format(sum(features)))

		#TODO: eval metric NDCG + overview params to be optimized
		param = {'max_depth':max_depth,
				 'eta':eta,
				 'silent':1,
				 'gamma':gamma,
				 'objective':'rank:pairwise',
				 'subsample':subsample,
				 'colsample':colsample,
				 'tree_method':'hist', 
				 #updater':'grow_gpu',
				 'eval_metric':'ndcg',
				 'predictor':'cpu_predictor',
				 'seed':42, 
				 'nthread':12}


		####
		# print('click_bool' in self.train_df.columns)
		# print('booking_bool' in self.train_df.columns)
		# print(self.train_df.columns)
		# input('hier')

		# Drop columns in function call to keep data but don't use it for training
		dtrain = self.get_DMatrix_from_data(self.train_df.drop(columns=['srch_id', 'prop_id']), self.train_weight, self.train_groups,features)
		dvalid = self.get_DMatrix_from_data(self.valid_df.drop(columns=['srch_id', 'prop_id']), self.valid_weight, self.valid_groups, features)


		evallist = [(dtrain, 'train'), (dvalid, 'eval')]

		num_round = 10000
		bst = xgb.train(param, dtrain, num_round, evals= evallist, early_stopping_rounds = 10)

		
		predictions = self.predict(bst, dvalid, self.valid_df, self.valid_weight)
		# predictions is df containing [srch_id, prop_id, pred_score, weight] ordered on [srch_id, pred_score]
		
		################################
		# Compute NDCG on predictions
		################################

		#Best NDCG score on valid set
		best_score = bst.best_score
		bst.__del__()
		# ^ its respective prediction
		#pbest_pred = bst.best_iteration

		dtrain.__del__()
		dvalid.__del__()

		del dtrain
		del dvalid
		del bst
		gc.collect()
		print(gc.collect())
		time.sleep(10)

		return best_score





	def __init__(self):
		file_train = 'datasets/GA_train'
		file_valid = 'datasets/GA_valid'
		self.train_df, self.train_weight, self.train_groups = self.get_DMatrix_data(file_train)
		self.valid_df, self.valid_weight, self.valid_groups = self.get_DMatrix_data(file_valid)

		

		#genotype = np.random.randint(0,2,152)
		#self.run_on_genotype(genotype)
		
