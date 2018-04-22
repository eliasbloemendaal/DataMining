import pandas as pd
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM, SimpleRNN
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime
import random



class lstm:

	def get_weight(self, bit):
		s = str()

		for elem in bit:
			s += str(elem)

		value = int(s, 2) / 100

		return value

	def run_on_genotype(self, genotype):
		df = self.df

		errors = []

		user_subset = random.sample(self.users, self.k)

		for user in user_subset:
			# train/test split
			data = df[df.id == user]

			# sort and shift
			data['mood'] = data['mood'].shift(-1)
			data = data[:-1]

			split_date = '2014-04-15'
			split = int(data.shape[0] * 2/3)
			train = data.iloc[:split, :].values
			test = data.iloc[split:, :].values

			train_x = np.delete(train, [28,29], axis=1)
			test_x = np.delete(test, [28,29], axis=1)
			train_y = train[:,29]
			test_y =test[:,29]


			deletion_vector = genotype[:37]
			weight_regularizer = self.get_weight(genotype[37:])

			# code to delete
			train_x = np.delete(train_x, genotype[:-1].nonzero(), axis=1)
			test_x = np.delete(test_x, genotype[:-1].nonzero(), axis=1)

			if genotype[-1] == 1:
				train_x = train_x[:, :-7]
				test_x = test_x[:, :-7]



			X_train = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
			X_test = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
			y_train = np.transpose(np.matrix(train_y))
			y_test = np.transpose(np.matrix(test_y))

			model = Sequential()
			model.add(LSTM(40, input_shape = (X_train.shape[1], X_train.shape[2]), activation='relu'))
			model.add(Dense(40, activation = 'relu'))
			model.add(Dense(40, activation = 'relu'))
			model.add(Dense(y_train.shape[1], activation = 'relu', W_regularizer=l2(weight_regularizer)))

			model.compile(loss='mean_squared_error', optimizer = 'adam')
			model.fit(X_train, y_train, epochs = 200, batch_size=100, verbose=2)

			Y_hat = model.predict(X_test)

			Y_hat = self.scaler_y.inverse_transform(Y_hat)
			y_test = self.scaler_y.inverse_transform(y_test)
			mse = mean_squared_error(Y_hat, y_test)

			# TODO:
			errors.append(mse)

		return errors





	def __init__(self):
		self.k = 5
		df = pd.read_csv('../assignment1/RuthsList.csv')


		self.users = set(df.id)

		# fill missing data
		for user in self.users:

			# mood
			df.loc[df.id == user, 'mood'] = df.loc[df.id == user, 'mood'].fillna(method='ffill')

			for col in df.columns:
				nr_nulls = df[df.id == user][col].isnull().sum()
				nr_rows_user = len(df[df.id == user])
				
				if nr_nulls > 0:

					if nr_rows_user - nr_nulls > 0:
						value = df[df.id == user][col].mean()
					else:
						# no obserations for attribute for this user
						value = df[col].mean()

					# print('setting for user {} col {} value {}'.format(user, col, value))
					df.loc[df.id == user, col] = df.loc[df.id == user, col].fillna(value)

		df['time'] = df['Unnamed: 0']
		del df['Unnamed: 0']
		df = df.set_index('time')

		# remove wrong dates
		df = df.loc[df.index>'2010-1-1']

		extra_data = pd.read_csv('extra_data.csv')
		extra_data['datetime'] = extra_data['datetime'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d'))
		extra_data.set_index('datetime', inplace=True)
		df = df.join(extra_data, how='left')
		

		# normalization
		cols = [col for col in df.columns if col!='id' and col!='mood']
		self.scaler_x = MinMaxScaler(feature_range=(0,1))
		self.scaler_y = MinMaxScaler(feature_range=(0,1))

		df[cols] = self.scaler_x.fit_transform(df[cols])
		mood = df['mood'].values.reshape(-1, 1)
		df['mood'] = self.scaler_y.fit_transform(mood)

		self.df = df