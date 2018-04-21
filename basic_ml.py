import pickle
import sklearn as sk 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
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
import os
#mpl.use('agg')
def do_ml(day):
    ################################################################
    # Modules to use
    ###############################################################
    USE_SAX = False
    FEATURE_REDUCTION = False
    DIM_REDUCTION_SEARCH = False

    #Create folder for administration
    try:
        os.mkdir('performance-{}-days'.format(day))
        os.mkdir('performance-{}-days/models'.format(day))
    except FileExistsError as e:
        None
    ################################################################


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
                doc = open('performance-{}-days/models/{}-{}-{}.pickle'.format(day, f.__name__,model_name, dim), 'wb')
                pickle.dump(optimized_model, doc)
                doc.close()

        adm_df = pd.DataFrame(rows_list)
        adm_df.to_csv('performance-{}-days/{}-dim_reduction.csv'.format(day, model_name))



    def dim_reduction_search(X, y):

        rf_params  = {"max_depth": [3, None],
                      "max_features": [1, 3, 10, 'sqrt', 'log2', 'auto'],
                      "min_samples_split": [2, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["mse"]}


        ada_params = {'n_estimators':[10, 50, 100, 300, 500],
                        'learning_rate':[1, 0.5, 0.1, 0.01, 0.001],
                        'loss':['linear', 'square', 'exponential']
                    }

        dims = [10, 20, 30]

        _find_best_dim_red(dims, AdaBoostRegressor(), 'AdaBoost', X, y, ada_params)
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
    if DIM_REDUCTION_SEARCH:
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


    params = {'RF': {"max_depth": [3, None],
                      "max_features": [1, 3, 10, 'sqrt', 'log2', 'auto'],
                      "min_samples_split": [2, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["mse"]},


        'AdaBoost' : {'n_estimators':[10, 50, 100, 300, 500],
                        'learning_rate':[1, 0.5, 0.1, 0.01, 0.001],
                        'loss':['linear', 'square', 'exponential']
                    }}

    ##############################################################
    # Recursive Feature Elimination
    #############################################################

    for estimator in [(AdaBoostRegressor(), 'AdaBoost'), (RandomForestRegressor(), 'RF')]:

        #Get features and store feature selector
        print('Optimizing {} using CV RFE'.format(estimator[1]))
        t0 = time.time()
        selector = RFECV(estimator[0], step=1, cv=10)
        selector = selector.fit(X_train, y_train)
        X_train_transformed = selector.transform(X_train)
        X_test_transformed = selector.transform(X_test)
        t1 = time.time()
        print('Optimizing features done in {} seconds, storing model..'.format(t1-t0))
        print('Selected features ({}): {}'.format(len(selector.get_support()[selector.get_support()]),selector.get_support()))
        doc = open('performance-{}-days/models/RFE-{}-selector.pickle'.format(day, estimator[1]), 'wb')
        pickle.dump(selector, doc)
        doc.close()

        #Optimize hyperparameters and evaluate model
        print('Start optimizing hyperparameters using determined features...')
        t0 = time.time()
        opt_model = GridSearchCV(estimator[0], params[estimator[1]], cv = 10, refit = True)
        opt_model.fit(X_train_transformed, y_train)
        t1 = time.time()

        print('Optimizing took {} seconds'.format((t1-t0)))
        print('best found parameters')
        print(opt_model.best_params_)
        y_pred = opt_model.predict(X_test_transformed)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        print("MSE on test set: {}".format(mse))
        model_doc = open('performance-{}-days/models/RFE-{}-model.pickle'.format(day,estimator[1]), 'wb')
        pickle.dump(opt_model, model_doc)
        model_doc.close()

    ##############################################################
    # Feature Stability Selection
    #############################################################   

    RL = RandomizedLasso(alpha='aic')
    print('Start optimizing using Randomized Lasso')
    t0 = time.time()
    RL.fit(X, y)
    t1 = time.time()
    print('Optimizing done in {} seconds'.format(t1-t0))
    print('Best parameters: {}'.format(RL.get_params()))
    print('Best features: {}'.format(RL.get_support()))
    doc = open('performance-{}-days/models/RandomizedLasso-selector.pickle'.format(day), 'wb')
    pickle.dump(RL, doc)
    doc.close()

    X_train_RL = RL.transform(X_train)
    X_test_RL = RL.transform(X_test)
    print('Using RL features to optimize model..')
    for estimator in [(AdaBoostRegressor(), 'AdaBoost'), (RandomForestRegressor(), 'RF')]:
        print('Optimizing {} using CV RFE'.format(estimator[0]))
        t0 = time.time()
        opt_model = GridSearchCV(estimator[0], params[estimator[1]], cv = 10, refit = True)
        opt_model.fit(X_train_RL, y_train)
        t1 = time.time()

        print('Optimizing took {} seconds'.format((t1-t0)))
        print('best found parameters')
        print(opt_model.best_params_)
        y_pred = opt_model.predict(X_test_RL)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        print("MSE on test set: {}".format(mse))
        model_doc = open('performance-{}-days/models/RandomizedLasso-{}-model.pickle'.format(day,estimator[1]), 'wb')
        pickle.dump(opt_model, model_doc)
        model_doc.close()










