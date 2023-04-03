import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit

from utils.utils import prepare_experiment, kappa
from models.GFC import GFC_triu_net_avg


feats_val = pd.read_csv('valence_features_200.csv', index_col=0)
feats_ar = pd.read_csv('arousal_features_200.csv', index_col=0)
feats_dom = pd.read_csv('dominance_features_200.csv', index_col=0)
feats_lik = pd.read_csv('liking_features_200.csv', index_col=0)

targets = pd.read_csv('za_klasifikaciju.csv', index_col=0)

targets = targets[['Valence', 'Arousal', 'Dominance', 'Liking']]
targets[targets < 4.5] = 0
targets[targets >= 4.5] = 1

experiment_config = prepare_experiment(folds=7, epochs_train=50, save_folder='GFC_triu_avg_128Hz')

num_class = 2

seed = 23
tf.random.set_seed(seed)

for c in ['Valence', 'Arousal', 'Dominance', 'Liking']:
    
    if c == 'Valence':
        data = normalize(feats_val, axis=0)
    elif c == 'Arousal':
        data = normalize(feats_ar, axis=0)
    elif c == 'Dominance':
        data = normalize(feats_dom, axis=0)
    elif c == 'Liking':
        data = normalize(feats_lik, axis=0)

    x_train, x_test, y_train, y_test = train_test_split(data, targets[c], test_size=0.2)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    clf = KerasClassifier(
        GFC_triu_net_avg,
        random_state=seed,
        
        #model hyperparameters
        nb_classes=num_class, 
        Chans = x_train.shape[0], 
        Samples = x_train.shape[1],
        dropoutRate=0.5,
        l1 = 0, l2 = 0,
        filters=2, maxnorm=2.0,maxnorm_last_layer=0.5,
        kernel_time_1=25,strid_filter_time_1= 1,
        bias_spatial = False,

        #model config
        verbose=0,
        batch_size=500, #full batch        
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer="adam",
        optimizer_learning__rate=0.1,
        metrics = ['accuracy'],
        epochs = experiment_config['epochs_train']
    )
    # search params
    param_grid =  {
                'filters':[2,4], #2,4
                'kernel_time_1':[25,50], #25,50
                }
    
    #Gridsearch
    scoring = {"AUC": 'roc_auc', "Accuracy": make_scorer(accuracy_score),'Kappa':make_scorer(kappa)}
    
    cv = GridSearchCV(clf,param_grid,cv=StratifiedShuffleSplit(n_splits = experiment_config['folds'], test_size = 0.2, random_state = seed),
                         verbose=0,n_jobs=1,
                         scoring=scoring,
                         refit="Accuracy",
                            )
    # frind best params with gridsearch
    cv.fit(x_train,y_train)
    
    # best score
    print('Accuracy',cv.best_score_)
    print('---------')
    
    cv.cv_results_['best_index_']=cv.best_index_
    
    #########
    cv.best_estimator_.model_.save_weights(f'{experiment_config["path"]}Model_{experiment_config["experiment"]}_{experiment_config["model_name"]}_'+experiment_config["data_set"]+'_4_40_weights.h5')
    with open(experiment_config["path"]+'Results_'+experiment_config["experiment"]+'_'+experiment_config["model_name"]+'_sujeto_'+'_'+experiment_config["data_set"]+'_4_40.p','wb') as f:
        pickle.dump(cv.cv_results_,f)     
