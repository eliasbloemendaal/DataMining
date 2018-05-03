import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import math

eps = 0.01

df = pd.read_csv('results.txt', header=-1)
gens = len(df)

df = df.iloc[:,:-1]

print(df.head())

def inv_transform(fitness):
	return 1/fitness - eps



mean_fitness = df.apply(lambda row: row.mean(), axis=1)
sd_fitness = df.apply(lambda row: row.std(), axis=1)

df = df.apply(lambda fitness: inv_transform(fitness), axis=0)

mean_error = df.apply(lambda row: row.mean(), axis=1)
sd_error = df.apply(lambda row: row.std(), axis=1)

results = pd.DataFrame({'mean_fitness':mean_fitness,
						'sd_fitness':sd_fitness,
						'mean_error':mean_error,
						'sd_error':sd_error})


results['upper_fitness'] = results['mean_fitness'] + 1.96 * results['sd_fitness'] / math.sqrt(10)
results['lower_fitness'] = results['mean_fitness'] - 1.96 * results['sd_fitness'] / math.sqrt(10)

results['upper_error'] = results['mean_error'] + 1.96 * results['sd_error'] / math.sqrt(10)
results['lower_error'] = results['mean_error'] - 1.96 * results['sd_error'] / math.sqrt(10)



# first plot
plt.subplot(1, 2, 1)
results['upper_fitness'].plot(color='lightblue', alpha=0.8)
results['lower_fitness'].plot(color='lightblue', alpha=0.8)
results['mean_fitness'].plot(color='blue')
plt.xlim(0,gens-1)

plt.xlabel('Generation')
plt.ylabel('Fitness')
# red_patch = mpatches.Patch(color='red', label='The red data')
mf = mlines.Line2D([], [], color='blue', markersize=15, label='Mean fitness')
ci = mlines.Line2D([], [], color='lightblue', markersize=15, label='95% confidence interval')
plt.legend(handles=[mf, ci])

plt.title('Evolution of fitness')


# second plot
plt.subplot(1, 2, 2)
results['upper_error'].plot(color='lightblue', alpha=0.8)
results['lower_error'].plot(color='lightblue', alpha=0.8)
results['mean_error'].plot(color='blue')
plt.xlim(0,gens-1)




plt.xlabel('Generation')
plt.ylabel('MSE')
mf = mlines.Line2D([], [], color='blue', markersize=15, label='Mean MSE')
ci = mlines.Line2D([], [], color='lightblue', markersize=15, label='95% confidence interval')
plt.legend(handles=[mf, ci])

plt.title('Evolution of mean MSE')


print(df)
plt.show()

