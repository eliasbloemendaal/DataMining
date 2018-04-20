import pickle
import sklearn as sk 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib as mpl
from math import sqrt
from copy import deepcopy
import pandas as pd
#mpl.use('agg')

USE_SAX = False
FEATURE_REDUCTION = False
print('Using sax: {}'.format(USE_SAX))
if USE_SAX:
    X = pickle.load(open('X_sax.p', 'rb'))
    y = pickle.load(open('y_sax.p', 'rb'))
else:
    X = pickle.load(open('X_reg.p', 'rb'))
    y = pickle.load(open('y_reg.p', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


################################################################
# Dimensionality reduction
################################################################

def pca_reduce(X, dim):
    pca = PCA(n_components = dim)

    X_reduced = pca.fit_transform(X)
    return X_reduced

def isomap_reduce(X, dim):
    iso = Isomap(n_components = dim)

    X_reduced = iso.fit_transform(X)
    return X_reduced

def _find_best_dim_red(dims, model, model_name, X, y, params):

    rows_list = []
    for dim in dims:
        for f in [pca_reduce, isomap_reduce]:

            print('Start reducing dimensionality using {} to {} dimensions'.format(f.__name__, dim))
            t0 = time.time()
            #reduce dimensionality
            print(X.shape)
            X_red = f(X, dim)
            print(X_red.shape)
            X_train_red, X_test_red,y_train, y_test = train_test_split(X_red, y, test_size = 0.2, random_state = 1234)
            X_train_red = f(X_train_red, dim)
            X_test_red = f(X_test_red, dim)
            t1 = time.time()
            print('Reducing dimensions cost {} seconds'.format(t1-t0))
            #Optimize model using grid search and cross validation
            print('Start optimizing {} model'.format(model_name))
            t0 = time.time()
            optimized_model = GridSearchCV(model, params, cv = 10, refit = True)
            optimized_model.fit(X_train_red, y_train)
            t1 = time.time()
            print('Optimizing took {} seconds'.format(t1-t0))
            print('best found parameters')
            print(optimized_model.best_params_)

            y_pred = optimized_model.predict(X_test_red)

            mse = sk.metrics.mean_squared_error(y_test, y_pred)
            print("MSE on test set: {}".format(mse))
            #administration
            rows_list.append({'model':deepcopy(model_name),
                                'dimensions':deepcopy(dim),
                                'reduction technique':deepcopy(f.__name__),
                                'mse':deepcopy(mse),
                                'parameters':str(deepcopy(optimized_model.best_params_))})
            #store model
            doc = open('models/{}-{}-{}.pickle'.format(f.__name__,model_name, dim), 'wb')
            pickle.dump(optimized_model, doc)
            doc.close()

    adm_df = pd.DataFrame(rows_list)
    adm_df.to_csv('{}-dim_reduction.csv'.format(model_name))

                
            
def dim_reduction_search(X, y):

    rf_params  = {"max_depth": [3, None],
                  "max_features": [1, 3, 10, 'sqrt', 'log2', 'auto'],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["mse", "mae"]}

    
    ada_params = {'n_estimators':[10, 50, 100, 300, 500],
                    'learning_rate':[1, 0.5, 0.1, 0.01, 0.001],
                    'loss':['linear', 'square', 'exponential']
                }

    dims = [ 10, 20, 30, 50]

    #_find_best_dim_red(dims, AdaBoostRegressor(), 'AdaBoost', X, y, ada_params)
    _find_best_dim_red(dims, RandomForestRegressor(), 'RandomForest', X, y, rf_params)



#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2)

##############################################################
# Feature selection
##############################################################

#Recursive Feature Elimination
def ReFeEl(nr_features, X_train, y_train, X_test, y_test, estimator, nr_models = 5):
    print('Start selecting features')
    t1 = time.time()
    #estimator = AdaBoostRegressor(learning_rate= 0.001, loss ='square', n_estimators  = 50)
    result = []
    for nr_feature in nr_features:
        selector = RFE(estimator, nr_feature, step=1)
        selector.fit(X_train, y_train)
        y_pred = selector.predict(X_test)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        result.append((mse, selector))
    #sort models and take nr_models best ones
    result.sort(key=lambda x: x[0])
    result = result[:nr_models]

    t2 = time.time()
    print('selecting features took {} seconds'.format(t2-t1))
    print(result[0][1].support_)
    print(result[0][1].ranking_)
    print('Minimum MSE: {}, number of selected features: {}'.format(result[0][0], len(result[0][1].support_[result[0][1].support_])))

    return result

if FEATURE_REDUCTION:
    if not USE_SAX:
        estimator = AdaBoostRegressor(learning_rate= 0.001, loss ='square', n_estimators  = 50)
    else:
        estimator = AdaBoostRegressor(learning_rate= 0.01, loss ='linear', n_estimators  = 50)
    _, total_nr_features = X.shape
    nr_features = range(1, total_nr_features)

    opt_features_models = ReFeEl(nr_features, X_train, y_train, X_test, y_test, estimator)
    print(opt_features_models)

    with open('optimal_features_model_{}.pickle'.format(USE_SAX), 'wb') as f:
        pickle.dump(opt_features_models, f)

# Feature Importance using Extra Trees
def ET_feature_selection():
    estimator = ExtraTreesRegressor()
    estimator.fit(X, y)
    print(estimator.feature_importances_)
    print((len(estimator.feature_importances_[estimator.feature_importances_ < 0.01]), len(estimator.feature_importances_)))
    print(np.mean(estimator.feature_importances_))
    print(np.std(estimator.feature_importances_))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot(estimator.feature_importances_)
    fig.savefig('boxplot.png', bbox_inches='tight')
    input('hallo')

##############################################################
#Dimensionality reduction search
##############################################################
dim_reduction_search(X, y)

##############################################################
#Grid search + CV
##############################################################
def append_deep_copy(rows_list, model_name, nr_features, mse, params):
    result = {'model':deepcopy(model_name),
                'nr_features':deepcopy(nr_features),
                'MSE':deepcopy(mse),
                'parameters':str(deepcopy(params))}
    rows_list.append(result)

    return rows_list


rows_list = []
opt_features_models = pickle.load(open('optimal_features_model_{}.pickle'.format(USE_SAX), 'rb'))
for opt_feature_model in opt_features_models:
    feature_set = opt_feature_model[1].support_
    nr_features = len(opt_feature_model[1].support_[opt_feature_model[1].support_])

    #reduce data set to chosen features
    red_X_train = X_train[:,feature_set]
    red_X_test = X_test[:,feature_set]

    print('Start CV grid search for {} features.'.format(nr_features)) 


    #AdaBoost
    t1 = time.time()
    print('start optimizing parameters for AdaBoostRegressor')
    ada_regr = AdaBoostRegressor()
    ada_params = {'n_estimators':[10, 50, 100, 300, 500],
                    'learning_rate':[1, 0.5, 0.1, 0.01, 0.001],
                    'loss':['linear', 'square', 'exponential']
                }

    ada_cv = GridSearchCV(ada_regr, ada_params, cv = 10, refit = True)
    ada_cv.fit(red_X_train, y_train)
    t2 = time.time()
    print('Optimizing took {} seconds'.format((t2-t1)))
    print('best found parameters')
    print(ada_cv.best_params_)

    

    ada_doc = open('models/Ada_regr-{}-{}.pickle'.format(USE_SAX, nr_features), 'wb')
    pickle.dump(ada_cv, ada_doc)
    ada_doc.close()
    #ada_cv.fit(red_X_train, y_train)
    y_pred = ada_cv.predict(red_X_test)

    mse = sk.metrics.mean_squared_error(y_test, y_pred)
    print("MSE on test set: {}".format(mse))

    #append data to rows_list
    rows_list = append_deep_copy(rows_list, 'Adaboost', nr_features, mse, ada_cv.best_params_)

    #Random Forest
    t3 = time.time()
    print('start optimizing parameters for RandomForestRegressor')
    rf_params  = {"max_depth": [3, None],
                  "max_features": [1, 3, 10, 'sqrt', 'log2', 'auto'],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["mse", "mae"]}

    rf_cv = GridSearchCV(RandomForestRegressor(), rf_params, cv = 10, refit = True)
    rf_cv.fit(red_X_train, y_train)
    t4 = time.time()
    print('Optimizing took {} seconds'.format((t4-t3)))
    print('best found parameters')
    print(rf_cv.best_params_)

    rf_doc = open('models/RF_regr{}-{}.pickle'.format(USE_SAX, nr_features), 'wb')
    pickle.dump(rf_cv, rf_doc)
    rf_doc.close()

    y_pred = rf_cv.predict(red_X_test)

    mse = sk.metrics.mean_squared_error(y_test, y_pred)
    print("MSE on test set: {}".format(mse))

    rows_list = append_deep_copy(rows_list, 'Random Forest', nr_features, mse, rf_cv.best_params_)


    '''#SVM
    t5 = time.time()
    print('start optimizing parameters for Support Vector Machine')
    svm_params  = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]

    svm_cv = GridSearchCV(SVR(), svm_params, cv = 10, refit = True)
    svm_cv.fit(red_X_train, y_train)
    t6 = time.time()
    print('Optimizing took {} seconds'.format((t6-t5)))
    print('best found parameters')
    print(svm_cv.best_params_)

    svm_doc = open('models/SVM_regr-{}-{}.pickle'.format(USE_SAX, nr_features), 'wb')
    pickle.dump(svm_cv, svm_doc)
    svm_doc.close()

    y_pred = svm_cv.predict(red_X_test)

    mse = sk.metrics.mean_squared_error(y_test, y_pred)
    print("MSE on test set: {}".format(mse))

    rows_list = append_deep_copy(rows_list, 'SVM', nr_features, mse, svm_cv.best_params_)'''

with open('rows_list-{}.pickle'.format(USE_SAX), 'wb') as f:
    pickle.dump(rows_list, f)
df = pd.DataFrame(rows_list)
df.to_csv('total_performance_overview-{}.csv'.format(USE_SAX))






