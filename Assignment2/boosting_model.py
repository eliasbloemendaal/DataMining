import xgboost as xgb
import math
import pandas as pd
import time
import gc
import os
import matplotlib.pyplot as plt
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
		# set groups and weights
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

	def  _chunkify(self, lst,n):
	    return [lst[i::n] for i in range(n)]

	def create_fold(self, fold_nr, K, features):
		
		#Create folds
		groups = self._chunkify(self.train_groups, K)

		# Get indices of desired fold in order to slice data
		start_index_valid = sum([sum(groups[i]) for i in range(fold_nr)])
		end_index_valid = start_index_valid + sum(groups[fold_nr])

		#create valid set
		valid_groups = groups.pop(fold_nr)
		valid_df = self.train_df.iloc[start_index_valid:end_index_valid,:]
		valid_weight = self.train_weight[start_index_valid:end_index_valid]

		dvalid = self.get_DMatrix_from_data(valid_df.drop(columns=['srch_id', 'prop_id']), valid_weight, valid_groups, features)

		# free up some memory
		# del valid_df
		# del valid_groups
		# del valid_weight

		# create train set
		train_groups = [item for sublist in groups for item in sublist] # flatten out chunkified list
		fold_train_df =  pd.concat([self.train_df.iloc[:start_index_valid,:], self.train_df.iloc[end_index_valid:,:]])
		#self.train_df[~self.train_df.isin(valid_df)]
		train_weight = self.train_weight[:start_index_valid].copy()
		train_weight.extend(self.train_weight[end_index_valid:])

		print('sum valid groups: {} valid shape {}, train shape {} overall shape {}'.format(sum(valid_groups), valid_df.shape, fold_train_df.shape, self.train_df.shape))
		dtrain = self.get_DMatrix_from_data(fold_train_df.drop(columns=['srch_id', 'prop_id']), train_weight, train_groups, features)

		return (dtrain, dvalid, valid_df, valid_weight)






	def cross_validate(self, genotype, K):
		"""
			Performs K-fold cross validation on genotype on train set
			Returns pd.DataFrame with evaluations
		"""

		# init parameters and features
		eta, gamma, max_depth, min_child_weight, subsample, colsample = self.get_params(genotype)
		features = self.get_features(genotype)
		# init list to store evaluations
		evaluations = []
		best_num_rounds = []
		
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

		num_round = 10000

		for fold in range(K):

			dtrain, dvalid, valid_df, valid_weight = self.create_fold(fold, K, features)
			evallist = [(dtrain, 'train'), (dvalid, 'eval')]
			bst = xgb.train(param, dtrain, num_round, evals= evallist, early_stopping_rounds = 10)
			predictions = self.predict(bst, dvalid, valid_df, valid_weight)

			best_num_rounds.append(bst.best_ntree_limit)
			#############
			# Compute NDCG and append to evaluations
			#############

		return evaluations

		# Return the evaluation history of the cross validation as a Pandas df


	def cross_val_genotypes(self, genotypes, K, folderpath):
		"""
			genotypes is a list of genotypes. Function performs K-fold cross validation on each genotype and writes results per genotype in folderpath
		"""
		filepath = os.path.join(folderpath,'geno_cross_validation.txt' )
		with open(filepath, a) as f:
			# cross validate each genotype
			for genotype in genotypes:
				evaluation = self.cross_validate(genotype, K)
				
				output_line = 'scores: {}, genotype {} \n'.format(evaluation, genotype)

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
		dtrain.__del__()
		dvalid.__del__()

		del dtrain
		del dvalid
		del bst
		gc.collect()

		return best_score


	def train_and_save(self, genotype, num_round, model_filepath):
		"""
		Function to train final model based on optimized genotype.
		self.train_df has to be based on  the full dataset 
		model is saved on model_filepath
		"""
		eta, gamma, max_depth, min_child_weight, subsample, colsample = self.get_params(genotype)
		features = self.get_features(genotype)		

		#TODO: eval metric NDCG + overview params to be optimized
		param = {'max_depth':max_depth,
				 'eta':eta,
				 'silent':0,
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


		# Drop columns in function call to keep data but don't use it for training
		dtrain = self.get_DMatrix_from_data(self.train_df.drop(columns=['srch_id', 'prop_id']), self.train_weight, self.train_groups,features)
		
		bst = xgb.train(param, dtrain, num_round)
		bst.save_model(model_filepath)

	def train_all_genotypes(self, genotypes, num_rounds, model_folder='models/'):
		"""
			function to train and save all genotypes on full dataset.
			self.train_df has to be constructed from the full train set
			function writes all model files to model_folder
		"""
		settings = zip(genotypes, num_rounds)

		for index, setting in enumerate(settings):
			genotype, num_round = setting
			model_filepath = os.path.join(model_folder, str(index)+'.model')

			self.train_and_save(genotype, num_round, model_filepath)


	def visualize_model(self, model_filepath):
		"""
			Visualizes one tree of a model for reporting purposes
		"""

		bst = xgb.Booster()
		bst.load_model(model_filepath)
		
		#plot the model
		xgb.plot_tree(bst, num_trees = 1, rank_dir = 'LR')
		plt.show()

	def predict_new_data(self, model_filepath, dtest):
		"""
			Function to load in saved model and predict unseen data
			dtest is loaded DMatrix for test data
		"""
		bst = xgb.Booster()
		bst.load_model(model_filepath)


		predictions = bst.predict(dtest)
		return predictions

	def ensemble_predict(self, model_folder, data_filepath, output_filepath):
		"""
			Makes prediction on test set with each model and writes away final ranking in output_filepath
		"""

		# load data
		df = pd.read_csv(data_filepath, header = 0)
		dtest = xgb.DMatrix(df)

		# make predictions using each model
		model_filenames = os.listdir(model_folder)
		for model_filename in model_filenames:
			df[model_filename] = self.predict_new_data(os.path.join(model_folder, model_filename), dtest)

		# Take average of all predictions
		df['pred_score'] = df[model_filenames].mean(axis=1)

		# Sort predictions and write to desired format
		df = df.sort_values(by=['srch_id', 'pred_score'], ascending=False, kind='heapsort')
		df[['srch_id','prop_id']].to_csv(output_filepath, index = None)


	def __init__(self):
		file_train = 'datasets/GA_train'
		file_valid = 'datasets/GA_valid'
		self.train_df, self.train_weight, self.train_groups = self.get_DMatrix_data(file_train)
		self.valid_df, self.valid_weight, self.valid_groups = self.get_DMatrix_data(file_valid)


		

		#genotype = np.random.randint(0,2,152)
		#self.run_on_genotype(genotype)
		
