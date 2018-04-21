import pandas as pd
from datetime import datetime
import holidays

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
print(temperatures)
dutch_holidays = holidays.CountryHoliday('NL')

df = pd.DataFrame({'datetime':datetimes, 'temperature': temperatures})
df['datetime'] = df['datetime'].apply(lambda t: datetime.strptime(t, '%Y%m%d'))
df['week_day'] = df.datetime.apply(lambda dt: dt.weekday())
df['holiday'] = df.datetime.apply(lambda dt: dt in dutch_holidays)

df = pd.get_dummies(df, columns = ['week_day'])





df.to_csv('extra_data.csv')