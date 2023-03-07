from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Code taken from https://github.com/nebojsa55/EmotionRecognition/blob/master/src/features_importance.ipynb

def read_feats(dir, n_patients):

    for i in range(1, n_patients):
        filename = "s{:02d}_{}.csv".format(i, (dir.split('_')[0]).lower())

        features_i = pd.read_csv(os.path.join(dir, filename), index_col=0)

        if i == 1:
            features = features_i.copy()
        else:
            features = features.append(features_i, ignore_index=True)
    
    return features

def get_mi(features, data, type_, feat_name):
    """
    INPUT
    ------
    features -> all features for one modality and one participant
    data -> classifiaction labels for all participants
    type_ -> str in ['Valence', 'Arousal', 'Dominance', 'Liking']
    feat_name -> name of modality

    RETURNS
    ------
    dict of feature names and their mutual info scores
    """

    if type_ == 'Valence':
        y = data['Valence'].copy()
    elif type_ == 'Arousal':
        y = data['Arousal'].copy()
    elif type_ == 'Dominance':
        y = data['Dominance'].copy()
    elif type_ == 'Liking':
        y = data['Liking'].copy()

    # Binary classification, 1-high, 0-low
    y[y < 4.5] = 0
    y[y >= 4.5] = 1

    try:
        m_i = mutual_info_classif(features, y)
    except ValueError:
        m_i = mutual_info_classif(features.fillna(0), y)

    return m_i

def get_feats(type_, names):

    feats = []
    for f in names:
        if f.split('-')[0] == type_:
            feats += [f.split('-', 1)[1]]

    return feats


if __name__ == "__main__":

    N_PATIENTS = 33
    data = pd.read_csv("za_klasifikaciju.csv")

    EEG_DIR = 'eeg_features/'
    electrodes = os.listdir(EEG_DIR)

    f_names_eeg = []

    mi_eeg = []

    for e in electrodes:

        for i in range(1, 33):

            filename = "s{:02d}_eegfeatures.csv".format(i)

            features_i = pd.read_csv(os.path.join(EEG_DIR, e, filename), index_col=0)

            if i == 1:
                features = features_i.copy()
            else:
                features = features.append(features_i, ignore_index=True)
                
        mi_e = []
        for t in ['Valence', 'Arousal', 'Dominance', 'Liking']:
            mi_e += [get_mi(features, data, t, e)]
        
        mi_eeg += [mi_e]

        f_names_electrode = [e+'-'+x for x in list(features)]
        f_names_eeg += f_names_electrode

    valence = np.array(mi_eeg[0])
    arousal = np.array(mi_eeg[1])
    dominance = np.array(mi_eeg[2])
    liking = np.array(mi_eeg[3])
    f_names_eeg = np.array(f_names_eeg)

    print(f_names_eeg.shape)

    #-------------------------------#

    ind_v = np.argsort(valence)
    valence = np.take_along_axis(valence, ind_v, 1)
    names_v = np.take_along_axis(f_names_eeg, ind_v, 0)

    ind_a = np.argsort(arousal)
    arousal = np.take_along_axis(arousal, ind_a, 1)
    names_a = np.take_along_axis(f_names_eeg, ind_a, 0)

    ind_d = np.argsort(dominance)
    dominance = np.take_along_axis(dominance, ind_d, 1)
    names_d = np.take_along_axis(f_names_eeg, ind_d, 0)

    ind_l = np.argsort(liking)
    liking = np.take_along_axis(liking, ind_l, 1)
    names_l = np.take_along_axis(f_names_eeg, ind_l, 0)
    
    #-------------------------------#

    valence = valence[-200:]
    names_v = names_v[-200:]

    arousal = arousal[-200:]
    names_a = names_a[-200:]

    dominance = dominance[-200:]
    names_d = names_d[-200:]

    liking = liking[-200:]
    names_l = names_l[-200:]

    #-------------------------------#
    
    valence_final = pd.DataFrame()
    arousal_final = pd.DataFrame()
    dominance_final = pd.DataFrame()
    liking_final = pd.DataFrame()


    for e in electrodes:
    # Valence
        eeg_feats_val = get_feats(e, names_v)

        # Arousal
        eeg_feats_ar = get_feats(e, names_a)

        # Dominance
        eeg_feats_dom = get_feats(e, names_d)

        # Liking
        eeg_feats_lik = get_feats(e, names_l)

        for i in range(1, 33):

            filename = "s{:02d}_eegfeatures.csv".format(i)

            features_i = pd.read_csv(os.path.join(EEG_DIR, e, filename), index_col=0)

            if i == 1:
                features = features_i.copy()
            else:
                features = features.append(features_i, ignore_index=True)

        valence_final = pd.concat([valence_final, features[eeg_feats_val]], axis=1)
        arousal_final = pd.concat([arousal_final, features[eeg_feats_ar]], axis=1)
        dominance_final = pd.concat([dominance_final, features[eeg_feats_dom]], axis=1)
        liking_final = pd.concat([liking_final, features[eeg_feats_lik]], axis=1)
    
    valence_final.to_csv('valence_features_200.csv')
    arousal_final.to_csv('arousal_features_200.csv')
    dominance_final.to_csv('dominance_features_200.csv')
    liking_final.to_csv('liking_features_200.csv')