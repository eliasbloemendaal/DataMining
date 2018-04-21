import pandas as pandas
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
        for technique in feature_overview[model]
            ind.append(model+'-'+technique)
            rows.append([int(x) for x in feature_overview[model][technique]])

    df = pd.DataFrame(rows, index = ind, columns = feature_names)
    print(df)




RFE_ada = pickle.load(open('C:\Users\\nvanderheijden\Desktop\Data Mining Techniques\DataMining\models\RFE-AdaBoost-selector.pickle', 'rb'))
RFE_RF = pickle.load(open('C:\Users\\nvanderheijden\Desktop\Data Mining Techniques\DataMining\models\RFE-RF-selector.pickle', 'rb'))

feature_overview = {'Gradient Boosting Machine': {'RFE':RFE_ada.get_support()
                                                    'RL':...},
                    'Random Forest': {'RFE':RFE_RF.get_support(),
                                        'RL': ...}}
