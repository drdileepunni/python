import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost

import copy

from data_preparation import process

def getNNModel(n_layers=128, activation='relu', dropout=1, optimizer='adam'):
    # Creating NN model
    nn_model = Sequential()
    nn_model.add(Dense(n_layers, activation=activation, input_shape=(24,)))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(n_layers, activation=activation))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(n_layers, activation=activation))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(n_layers, activation=activation))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(n_layers, activation=activation))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(n_layers, kernel_regularizer=regularizers.l2(0.1), activation=activation))
    nn_model.add(Dense(1))
    # Compiling
    nn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return nn_model

def trainModels(pred_train, targ_train):

    # Creating pipes
    lr = LogisticRegression(random_state=42)
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor()
    sgb = GradientBoostingRegressor()
    xgb = xgboost.XGBRegressor()

    # Creating params
    params_lr = {
        'C' : [0.001, 0.01, 0.1, 1],
        'penalty' : ['l1', 'l2']
    }
    params_rf = {
        'max_depth' : np.arange(2, 20, 6),
        'max_leaf_nodes': np.arange(2,20,6), 
        'min_impurity_split': [0.01, 0.1],
        'n_estimators': np.arange(50, 70, 10)
    }
    params_gb = {
        'n_estimators': np.arange(10, 100, 10),
        'learning_rate': [0.01, 0.1, 1], 
        'max_depth': np.arange(1,10,1)
    }
    params_sgb = {
        'n_estimators': np.arange(10, 100, 10),
        'learning_rate': [0.01, 0.1, 1], 
        'max_depth': np.arange(1,10,1), 
        'subsample': [0.2, 0.4, 0.8], 
        'max_features': [0.2, 0.4, 0.8]
    }
    params_xgb = {
        'subsample': np.arange(.05, 1, .05),
        'max_depth': np.arange(3,20,1),
        'colsample_bytree': np.arange(.1,1.05,.05) 
    }


    # Construct grid searches
    jobs = -1

    grid_lr = RandomizedSearchCV(estimator=lr, 
                        param_distributions=params_lr,
                        n_iter=5,
                        scoring='roc_auc',
                        cv=5)
    grid_rf = RandomizedSearchCV(estimator=rf, 
                        param_distributions=params_rf, 
                        n_iter=5,
                        scoring='roc_auc', 
                        cv=5)
    grid_gb = RandomizedSearchCV(estimator=gb, 
                        param_distributions=params_gb, 
                        n_iter=5,
                        scoring='roc_auc', 
                        cv=5)
    grid_sgb = RandomizedSearchCV(estimator=sgb, 
                        param_distributions=params_sgb, 
                        n_iter=5,
                        scoring='roc_auc', 
                        cv=5)
    grid_xgb = RandomizedSearchCV(estimator=xgb, 
                        param_distributions=params_xgb, 
                        n_iter=5,
                        scoring='roc_auc', 
                        cv=5)

    # List of pipelines
    grids = [grid_sgb, grid_gb, grid_rf, grid_lr, grid_xgb, 1]

    # Dictionary of pipelines
    grid_dict = {0: 'Stochastic Gradient Boost', 1: 'Gradient Boost', 2: 'RandomForestRegressor', 
                3: 'Logistic Regression', 4: 'XGBoost', 5: 'Neural Net'}

    # Fitting
    print('Performing model optimizations...')

    X_train, X_test, y_train, y_test = train_test_split(pred_train, targ_train, test_size=0.3, stratify=targ_train, random_state=62)

    for idx, gs in enumerate(grids):
        if (grid_dict[idx] == 'Neural Net'):
            print('\nEstimator: %s' % grid_dict[idx])
                        
            nn_model = getNNModel()
            nn_model.fit(pred_train, targ_train)
            y_pred_nn = nn_model.predict(pred_train)
            # Scoring
            print('Train set accuracy score for best param: %.3f ' % roc_auc_score(targ_train, y_pred_nn))
            break
        else:
            print('\nEstimator: %s' % grid_dict[idx])
            # Fit grid search
            gs.fit(X_train, y_train)
            # Best params
            print('Best params: %s' % gs.best_params_)
            # Best training data accuracy
            print('Best training accuracy: %.3f' % gs.best_score_)
            if (grid_dict[idx] == 'Logistic Regression'):
                y_pred = gs.predict_proba(X_test)[:, 1]
            else:
                # Predict on test data with best params
                y_pred = gs.predict(X_test)
            # Test data accuracy of model with best params
            print('Test set accuracy score for best params: %.3f ' % roc_auc_score(y_test, y_pred))
    return [grid_sgb.best_estimator_, grid_gb.best_estimator_, grid_rf.best_estimator_, grid_lr.best_estimator_, grid_xgb.best_estimator_, nn_model]

