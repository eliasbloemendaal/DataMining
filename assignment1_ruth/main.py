import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

df = pd.read_csv('dataset_mood_smartphone.csv')
df.time = df.time.apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f'))
# print(data.head(100))

# print(data.groupby('variable')['value'].mean())
users = set(df.id.tolist())

actual = []
predictions = []

# print(df)
moods = df[df.variable=='mood']

for user in users:

	# predictions
	s08 = df[(df.id == user) & (df.variable == 'mood')].resample('D', on='time').mean()
	s08 = s08.fillna(method='ffill')
	s08['prediction'] = s08.value.shift(1)
	s08 = s08.dropna()
	# print(s08)
	s08[['value', 'prediction']].plot()
	# plt.show()
	actual.extend(s08.value.tolist())
	predictions.extend(s08.prediction.tolist())

actual = np.array(actual)
predictions = np.array(predictions)

print(actual, predictions)

# calculate MSE
MSE = np.square(actual - predictions).mean()

print(MSE)

