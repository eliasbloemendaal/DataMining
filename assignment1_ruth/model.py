import pandas as pd
import keras
from keras.layers import Input, Dense, LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('../DataMining/RuthsList.csv')

users = set(df.id)

# fill missing data
for user in users:

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

# normalization


# train/test split


print(df.head())



model = Sequential()

model.add(LSTM((1), batch_input_shape=(None, 5,1), return_sequences=True))
model.add(Dense())
# model.compile(loss='mean_absolute_error', optimizer='adam', metric=['accuracy'])
# model.summary()


# history = model.fit()

# mood = Input(shape=(datalen, 1), name='mood')
# mood_layers  = LSTM(64, return_sequences=False)(mood)
# output = Dense(labellen, activation='linear', name='mood_hat')(output)

# model = Model(inputs = mood, outputs = output)
# model.compile(optimizer='rmsprop', loss='mse')
