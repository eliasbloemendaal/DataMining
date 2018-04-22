import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_rel
from matplotlib.ticker import MultipleLocator

def visualize_features(feature_names, feature_overview):
    """
    {model: {technique:[features]}}
    Technique is a (technique_name, selector) tuple.
    Selector is a list of True/False values telling whether the feature at index i is used in the model
    """
    ind = []
    rows = []
    print(feature_overview)
    input('hier')
    for model in feature_overview.keys():
        for technique in feature_overview[model].keys():
            ind.append(model+'-'+technique)
            rows.append([int(x) for x in feature_overview[model][technique]])

    df = pd.DataFrame(rows, index = ind, columns = feature_names)
    df = df.transpose()
    print(df.head())
    df.to_csv('SelectedFeatures.csv')
    vals = df.values

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

    the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, 
                        colWidths = [0.03]*vals.shape[1], loc='center', 
                        cellColours=plt.cm.RdYlGn(vals))
    plt.show()
    print('hier')

df = pd.read_csv('RuthsList-extended.csv', header = 0, index_col = 0)
labels = df.columns

RFE_ada = pickle.load(open('performance-4-days\models\RFE-AdaBoost-selector.pickle', 'rb'))
RFE_RF = pickle.load(open('performance-4-days\models\RFE-RF-selector.pickle', 'rb'))
print('hier: {}'.format(RFE_RF.get_params()))
RL = pickle.load(open('performance-4-days\models\RandomizedLasso-selector.pickle', 'rb'))

feature_overview = {'Gradient Boosting Machine': {'RFE':RFE_ada.get_support(),
                                                    'RL':RL.get_support()},
                    'Random Forest': {'RFE':RFE_RF.get_support(),
                                        'RL': RL.get_support()}}

#visualize_features(labels, feature_overview)

def combine_results(days, t_test):
    df_list = []

    for day in days:
        for model in ['AdaBoost', 'RandomForest']:
            filename = 'performance-{}-days/{}-dim_reduction.csv'.format(day, model)
            df = pd.read_csv(filename, header = 0, index_col = 0)
            df['day'] = day - 1
            df_list.append(df)

    df = pd.concat(df_list)
    df = df.drop(columns=['parameters'])
    if t_test:
        ada = df.mse[df.model == 'AdaBoost'].tolist()
        rf = df.mse[df.model == 'RandomForest'].tolist()

        ada.extend([0.2324, 0.2432, 0.2120, 0.2465,0.2671, 0.2231, 0.2385, 0.2097,0.2400,0.2750])
        rf.extend([0.2172,0.2529,0.2005,0.2411,0.2714,0.2216,0.2432,0.1984,0.2255,0.2752])

        print(ttest_rel(ada, rf))
    else:
        print(df.head())
        df = df.set_index(['day','dimensions'])
        #print(df.head(15))
        df1 = pd.DataFrame()
        df1['AdaBoost PCA'] = df.mse[(df.model == 'AdaBoost') & (df['reduction technique'] == 'pca_reduce')]
        df1['AdaBoost Isomap'] = df.mse[(df.model == 'AdaBoost') & (df['reduction technique'] == 'isomap_reduce')]
        df1['Random Forest PCA'] = df.mse[(df.model == 'RandomForest') & (df['reduction technique'] == 'pca_reduce')]
        df1['Random Forest Isomap'] = df.mse[(df.model == 'RandomForest') & (df['reduction technique'] == 'isomap_reduce')]

        df1.to_excel('DimensionReductionSummary.xlsx')

def plot_best_dim_reduction_day(dims, ada_pca, ada_iso, rf_pca, rf_iso, day, benchmark):

    ml = MultipleLocator(5)
    plt.plot(dims, ada_pca)
    plt.plot(dims, ada_iso)
    plt.plot(dims, rf_pca)
    plt.plot(dims, rf_iso)
    #plt.axhline(y=benchmark, color='r')
    plt.xticks([1, 2, 3, 4, 5])
    plt.axes().yaxis.set_minor_locator(ml)
    plt.axes().yaxis.set_tick_params(which='minor', right = 'off')
    plt.xlabel('Number days considered in history')
    plt.ylabel('MSE on test set')
    plt.title('Performance for feature selection techniques')
    #plt.legend(['AdaBoost Randomized Lasso','AdaBoost RFE', 'Random Forest Randomized Lasso', 'Random Forest RFE', 'Benchmark'])
    plt.legend(['AdaBoost Randomized Lasso','AdaBoost RFE', 'Random Forest Randomized Lasso', 'Random Forest RFE'])
    plt.show()



days = [2,3,4,5,6]
#combine_results(days, True)
day = 3
dims = [1,2,3,4,5]
ada_rl = [0.2324, 0.2432, 0.2120, 0.2465, 0.2671]
ada_rfe = [0.2231, 0.2385, 0.2097, 0.2400, 0.2750]
rf_rl = [0.2172, 0.2529, 0.2005, 0.2411, 0.2714]
rf_rfe = [0.2216, 0.2432, 0.184, 0.2255, 0.2752]
benchmark = 0.544


plot_best_dim_reduction_day(dims, ada_rl, ada_rfe, rf_rl, rf_rfe, day, benchmark)

ada_rl.extend(rf_rl)
ada_rfe.extend(rf_rfe)

print(ttest_rel(ada_rl, ada_rfe))
