import pandas as pd
from datetime import datetime

datetimes = []
temperatures = []

with open('KNMI_20140609.txt', 'r') as f:
	for line in f:
		if not line.startswith('#'):

			line = line.split(',')

			date_time = line[1]
			temperature = float(line[2].strip())/10
			datetimes.append(date_time)
			temperatures.append(temperature)
			

# print(datetimes)
# print(temperatures)

df = pd.DataFrame({'datetime':datetimes, 'temperature': temperatures})
df['datetime'] = df['datetime'].apply(lambda t: datetime.strptime(t, '%Y%m%d'))

df.to_csv('daily_temperatures.csv')