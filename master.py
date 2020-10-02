# import pandas as pd
# from sklearn import metrics
import pandas as pd
import numpy as np
import os
import sklearn
# from sklearn.model_selection import cross_val_score
from sklearn import metrics
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def init():
    ####################
    # IMPORT DATASET
    ####################
    os.listdir(os.getcwd())
    df = pd.read_csv("./student-por-new.csv", sep=";")
    # print(df.head())
    # print(df.columns)
    # df.info()

    return df


def preproc(df):
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

    x = db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
            'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
            'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
            'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
            'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
            'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
            'guardian_mother', 'guardian_other', 'G1', 'G2']]

    y = db['binary']

    return x, y

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

def select_numerical(x):
    print('Selecting only numerical attributes')
    x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]

    return x_num;

def PCA_study(x):
    print('Applying PCA')
    x_new = np.array(x)
    x = sklearn.preprocessing.scale(x_new)
    #print(len(x[0]))
    pca = sklearn.decomposition.PCA(n_components=45)
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

    xrange = range(1,46)
    plt.scatter(xrange, cum_var)
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
