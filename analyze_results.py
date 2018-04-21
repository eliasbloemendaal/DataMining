import pandas as pd
import matplotlib.pyplot as plt
import pickle

def visualize_features(feature_names, feature_overview):
    """
    {model: {technique:[features]}}
    Technique is a (technique_name, selector) tuple.
    Selector is a list of True/False values telling whether the feature at index i is used in the model
    """
    ind = []
    rows = []
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

RFE_ada = pickle.load(open('models\RFE-AdaBoost-selector.pickle', 'rb'))
RFE_RF = pickle.load(open('models\RFE-RF-selector.pickle', 'rb'))
RL = pickle.load(open('models\RandomizedLasso-selector.pickle', 'rb'))

feature_overview = {'Gradient Boosting Machine': {'RFE':RFE_ada.get_support(),
                                                    'RL':RL.get_support()},
                    'Random Forest': {'RFE':RFE_RF.get_support(),
                                        'RL': RL.get_support()}}

visualize_features(labels, feature_overview)
