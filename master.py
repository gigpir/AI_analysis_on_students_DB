# import pandas as pd
# from sklearn import metrics
from collections import Counter

import pandas as pd
import numpy as np
import sys
import os
import sklearn
import imblearn
# from sklearn.model_selection import cross_val_score
from sklearn import metrics
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt




def init(dataframe='por'):
    ####################
    # IMPORT DATASET
    ####################
    if dataframe =='por':
        df = pd.read_csv("./student-por.csv", sep=";")
    elif dataframe =='mat':
        df = pd.read_csv("./student-mat.csv", sep=";")
    else:
        print("Input errato: usare dataframe='por' o dataframe='mat'")
        sys.exit()
    # print(df.head())
    # print(df.columns)
    # df.info()
    return df

def label_pass(row):
        if row['G3'] >= 10:
            return 'pass'
        else:
            return 'fail'

def label_5_class(row):
        if row['G3'] < 10:
            return 'fail'
        elif row['G3'] < 12:
            return 'sufficient'
        elif row['G3'] < 14:
            return 'satisfactory'
        elif row['G3'] < 16:
            return 'good'
        else:
            return 'excellent'

def preproc(df, select='all'):
    ###################################
    # CREATE BINARY AND 5-CLASS COLUMNS
    ###################################

    #print(df.columns)
    #print(df.shape)
    df['binary']= df.apply(lambda row: label_pass(row), axis=1)
    df['5-class']= df.apply(lambda row: label_5_class(row), axis=1)
    #print(df.columns)
    #print(df.shape)
    #print(df.head)
    #counts = df['binary'].value_counts().to_dict()
    #print(counts)

    ####################
    # PREPROCESSING
    ####################
    # Create a map to perform mapping of binary attributes ONLY

    replace_binary_attributes_map = {'school': {'GP': 0, 'MS': 1},
                                     # school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
                                     'sex': {'F': 0, 'M': 1},
                                     # sex - student's sex (binary: 'F' - female or 'M' - male)
                                     'address': {'U': 0, 'R': 1},
                                     # address - student's home address type (binary: 'U' - urban or 'R' - rural)
                                     'famsize': {'LE3': 0, 'GT3': 1},
                                     # famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
                                     'Pstatus': {'T': 0, 'A': 1},
                                     # Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
                                     'schoolsup': {'yes': 1, 'no': 0},
                                     # schoolsup - extra educational support (binary: yes or no)
                                     'famsup': {'yes': 1, 'no': 0},
                                     # famsup - family educational support (binary: yes or no)
                                     'paid': {'yes': 1, 'no': 0},
                                     # paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
                                     'activities': {'yes': 1, 'no': 0},
                                     # activities - extra-curricular activities (binary: yes or no)
                                     'nursery': {'yes': 1, 'no': 0},
                                     # nursery - attended nursery school (binary: yes or no)
                                     'higher': {'yes': 1, 'no': 0},
                                     # higher - wants to take higher education (binary: yes or no)
                                     'internet': {'yes': 1, 'no': 0},
                                     # internet - Internet access at home (binary: yes or no)
                                     'romantic': {'yes': 1, 'no': 0},
                                     # romantic - with a romantic relationship (binary: yes or no)
                                     'binary': {'pass': 1, 'fail': 0},  # “pass” if G3>=10 else “fail”
                                     }
    df_2 = df.copy()

    df_2.replace(replace_binary_attributes_map, inplace=True)

    pd.set_option('display.max_columns', None)
    # print(df.head())
    # print(df_2.head())
    # print(df_2['Mjob'].value_counts())

    # Encode categorical data into new binary labels
    df_2 = pd.get_dummies(df_2, columns=['Mjob'])
    df_2 = pd.get_dummies(df_2, columns=['Fjob'])
    df_2 = pd.get_dummies(df_2, columns=['reason'])
    df_2 = pd.get_dummies(df_2, columns=['guardian'])

    # print(df_2.head())

    # print(df_2.columns)

    # Re order the table
    db = df_2[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
               'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
               'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
               'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
               'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
               'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
               'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
               'guardian_mother', 'guardian_other', 'G1', 'G2',
               'G3', 'binary', '5-class']]

    # print(db.head())
    if select=='all':
        x = db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
                'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
                'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
                'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
                'guardian_mother', 'guardian_other', 'G1', 'G2']]
    elif select=='G1':
        x = db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
                'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
                'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
                'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
                'guardian_mother', 'guardian_other', 'G1']]
    elif select=='G2':
        x = db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
                'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
                'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
                'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
                'guardian_mother', 'guardian_other', 'G2']]
    elif select=='novotes':
        x = db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
                'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
                'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
                'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
                'guardian_mother', 'guardian_other']]
    else:
        print("Invalid value of select, valid values are: 'all', 'G1', 'G2', 'novotes'")
        sys.exit()

    y = db['binary']
    feature_names= x.columns
    return x, y, feature_names

def split(x, y, scaled=False):
    x_new = np.array(x)
    y = np.array(y)
    y = np.ravel(y)
    if scaled==True:
        x = sklearn.preprocessing.scale(x_new)
    # divisione fra train e test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=123)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    return x_train, x_test, y_train, y_test

def select_numerical(x, select='all'):
    print('Selecting only numerical attributes')
    if select=='all':
        x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
    elif select=='G1':
        x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1']]
    elif select=='G2':
        x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','G2']]
    elif select=='novotes':
        x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
                'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']]
    return x_num;

def PCA_study(x,feature_names):
    print('Applying PCA')
    x_new = np.array(x)
    x = sklearn.preprocessing.scale(x_new)
    #print(len(x[0]))
    pca = sklearn.decomposition.PCA(n_components=len(feature_names))
    pca.fit(x)
    var=np.array(pca.explained_variance_ratio_)
    cum_var=np.cumsum(var)
    #print(len(cum_var))
    """
    for i in range(0,45):
        print('La varianza spiegata per {} componenti è {}'.format(i+1,cum_var[i]))
    """
    #24 componenti -> 0.84
    #27 componenti -> 0.86
    #30 componenti -> 0.90
    #38 componenti -> 0.99

    xrange = range(1,len(feature_names)+1)
    plt.scatter(xrange, cum_var)
    plt.bar(xrange, var)
    plt.grid(axis='y')
    plt.xlabel('Components of PCA')
    plt.ylabel('Variance explained')
    plt.title('Cumulative variance explained for different numbers of components of PCA')
    plt.xticks(np.arange(0, 47, 2))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()

def PCA(x,components):
    x_new = np.array(x)
    x = sklearn.preprocessing.scale(x_new)
    pca = sklearn.decomposition.PCA(n_components=components)
    x_pca=pca.fit_transform(x)
    return x_pca

def SMOTE(x,y): #provare anche categoric 2
    categoric1=[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    categoric2=[0,1,3,4,5,11,12,13,14,15,16,17,18,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
    smote_nc= imblearn.over_sampling.SMOTENC(categorical_features=categoric1,random_state=0)
    X_resampled, y_resampled= smote_nc.fit_resample(x,y)
    #print('Dataset after resampling:')
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled,y_resampled

def cv_SMOTE(model, X, y):
    X = np.array(X)
    y = np.array(y)
    y = np.ravel(y)
    kf = KFold(n_splits=10)
    score=[]
    for (train_index, test_index) in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        y_train= np.ravel(y_train)
        X_test = X[test_index]
        y_test = y[test_index]
        y_test= np.ravel(y_test)
        X_train_oversampled, y_train_oversampled = SMOTE(X_train, y_train)
        model.fit(X_train_oversampled, y_train_oversampled)
        y_pred = model.predict(X_test)
        score.append(model.score(X_test,y_test))
    return np.mean(score)