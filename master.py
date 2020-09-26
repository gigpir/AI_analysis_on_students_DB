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


# print(x)
# print(y)

def kNN(x, y):
    ####################
    # KNN
    ####################

    # KNN doesn't work well on datasets with many features and in particular with
    # sparse matrices (our case because we have a lot of categorical data)
    # https://medium.com/cracking-the-data-science-interview/k-nearest-neighbors-who-are-close-to-you-19df59b97e7d

    # Potrebbe essere una buona idea cercare di selezionare solo gli attributi numerici e togliere gli attributi categorici

    # trasformo il mio dataset in un array numpy
    x_new = np.array(x)
    y = np.array(y)
    y = np.ravel(y)

    # preprocessing -> standardizzazione
    x_scaled = sklearn.preprocessing.scale(x_new)

    # print(x_scaled)
    # print(y.shape)
    # for i in range(0,5):
    #    neigh = KNeighborsClassifier(n_neighbors=i)
    #   scores[i] = cross_val_score(neigh, x_scaled, y, cv=10)
    # print(scores)

    # divisione fra train e test set
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.5, random_state=123)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    score_train = []
    score_test = []
    f1_train = []
    f1_test = []
    score_cross = []
    # applico KNN con diversi parametri -> faccio da 1 a 11 a passi di 2 perchè deve essere dispari e dopo
    # una certa soglia si stabilizza e non è più interessante
    # calcolo la 10-fold cross validation accuracy
    for i in range(1, 21, 2):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(x_train, y_train)

        score_cross.append(np.mean(cross_val_score(neigh, x_scaled, y, cv=10)))

        # calcolo accuratezza del train e del test set -> studio overfitting
        y_pred_train = neigh.predict(x_train)
        y_pred_test = neigh.predict(x_test)

        score_train.append(metrics.accuracy_score(y_pred_train, y_train))
        score_test.append(metrics.accuracy_score(y_pred_test, y_test))

        f1_train.append(metrics.f1_score(y_pred_train, y_train))
        f1_test.append(metrics.f1_score(y_pred_test, y_test))

    # print(score_train)
    # print(score_test)
    # print(score_cross)


    #VALUTARE SE FARE UN SOLO PLOT, VEDI LOGISTIC REGRESSION -> viene abbastanza ordinato

    # Plot of train and test error for different values of K
    xrange = range(1, 21, 2)
    error_train = np.ones(len(xrange)) - score_train
    error_test = np.ones(len(xrange)) - score_test
    # error_cross= np.ones(len(xrange)) - score_cross
    plt.plot(xrange, error_train, label="train accuracy")
    plt.plot(xrange, error_test, label="test accuracy")
    # plt.plot(xrange, error_cross, label = 'cross val accuracy') if I want also cv error
    plt.xlabel('K')
    plt.ylabel('Score accuracy')
    plt.title('KNN train and test score error for different values of K')
    plt.legend()
    plt.show()

    # plot F1 error
    xrange = range(1, 21, 2)
    f1error_train = np.ones(len(xrange)) - f1_train
    f1error_test = np.ones(len(xrange)) - f1_test
    # error_cross= np.ones(len(xrange)) - score_cross
    plt.plot(xrange, f1error_train, label="train accuracy")
    plt.plot(xrange, f1error_test, label="test accuracy")
    # plt.plot(xrange, error_cross, label = 'cross val accuracy') if I want also cv error
    plt.xlabel('K')
    plt.ylabel('F1 Accuracy')
    plt.title('KNN train and test F1 error for different values of K')
    plt.legend()
    plt.show()
    #print("F1 test error is:")
    #print(f1error_test)

    # la scelta migliore con train-test è k=5 -> test_accuracy=0.86769 (val)

    # ATTENZIONE-> non dovrei scegliere il parametro migliore sulla base del test set, dovrei avere
    # train-val-test. In questo caso il nostro test set può essere considerato come validation e considerare
    # come stimatore dell'accuracy la cross validation

    # rialleniamo KNN per visualizzare la matrice di confusione
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)
    cv_accuracy=np.mean(cross_val_score(neigh, x_scaled, y, cv=10))
    matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    print("Confusion matrix for k=5 normalized by true categories (rows):")
    print(matrix)
    print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)

    # Bisognerebbe provare ad usare solo le variabili numeriche


def select_numerical(x):
    x_num = x[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
            'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]

    return x_num;

def logistic_regression(x,y):
    #nella logistic_regression non c'è bisogno di fare nessuna operazione di standardizzazione

    xnp = np.array(x)
    y = np.array(y)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(xnp, y, test_size=0.5, random_state=123)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    score_train = []
    score_test = []
    f1_train = []
    f1_test = []

    #Prima ho fatto una gridsearch, ma la norma L1 ha risultati sempre peggiori
    #search for better penalization parameter
    for c in [0.001, 0.01,0.1,1,10,100]:
        clf=sklearn.linear_model.LogisticRegression(C=c,max_iter=1000).fit(x_train, y_train)
        y_pred_train=clf.predict(x_train)
        y_pred_test=clf.predict(x_test)
        score_train.append(metrics.accuracy_score(y_pred_train, y_train))
        score_test.append(metrics.accuracy_score(y_pred_test, y_test))
        f1_train.append(metrics.f1_score(y_pred_train,y_train))
        f1_test.append(metrics.f1_score(y_pred_test,y_test))
        #print("Report per fattore di penalizzazione:",c)                    #fighissimo ma stampa un sacco di cose
        #print(sklearn.metrics.classification_report(y_pred_test, y_test))   #dopo commentalo
    """
    print(score_train)
    print(f1_train)
    print(score_test)
    print(f1_test)
    """

    #CHIEDERE A GIGI SE PREFERISCE QUESTO GRAFICO OPPURE DUE COME KNN
    xrange = [0.001, 0.01,0.1,1,10,100]
    f1error_train = np.ones(len(xrange)) - f1_train
    f1error_test = np.ones(len(xrange)) - f1_test
    error_train = np.ones(len(xrange)) - score_train
    error_test = np.ones(len(xrange)) - score_test
    plt.plot(xrange, error_train, label="train score accuracy", color='navy')
    plt.plot(xrange, error_test, label="test score accuracy", color='green')
    plt.plot(xrange, f1error_train, label="train F1 accuracy", color='royalblue')
    plt.plot(xrange, f1error_test, label="test F1 accuracy", color='lightgreen')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression train and test F1 and score accuracy for different C')
    plt.legend()
    plt.show()


    #il fattore di penalizzazione migliore è 1, alleniamo di nuovo con questo parametro e calcoliamo la CV
    clf=sklearn.linear_model.LogisticRegression(C=1,max_iter=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
    matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    print("Confusion matrix for k=5 normalized by true categories (rows):")
    print(matrix)
    print("10-fold cross validation accuracy for C=1 is:", cv_accuracy)
    print("Report del test set per fattore di penalizzazione=", 1)
    print(sklearn.metrics.classification_report(y_pred, y_test))

    #direi di fare LDA e SVM ora che sono tutti classificatori lineari

def LDA(x,y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    xnp = np.array(x)
    y = np.array(y)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(xnp, y, test_size=0.5, random_state=123)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    clf = LinearDiscriminantAnalysis() #ho provato ad usare lo shrinkage con un solver diverso -> forma di regolarizzazione
    clf.fit(x_train, y_train)           #ma l'accuracy è più o meno la stessa e così non dobbiamo spiegarla nel report :D
    y_pred_test = clf.predict(x_test)
    score_test=metrics.accuracy_score(y_pred_test, y_test)
    f1_test=metrics.f1_score(y_pred_test, y_test)
    print('Test score accuracy is:', score_test)
    print('Test f1 accuracy is:', f1_test)
    print("Report:")
    print(sklearn.metrics.classification_report(y_pred_test, y_test))

    cv_accuracy = np.mean(cross_val_score(clf, x, y, cv=10))
    matrix = metrics.confusion_matrix(y_test, y_pred_test, normalize="true")
    print("Confusion matrix for k=5 normalized by true categories (rows):")
    print(matrix)
    print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)

def SVD(x,y):

    xnp = np.array(x)
    y = np.array(y)
    y = np.ravel(y)
    x_train, x_test, y_train, y_test = train_test_split(xnp, y, test_size=0.5, random_state=123)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    #normalizzazione??


    #script per classi non bilanciate -> da provare e capire se si può applicare anche agli altri
    #https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py

    #devo fare un for per il parametro di regolarizzazione
    #posso provare diversi kernel


    #RBF KERNEL
    score_train = []
    score_test = []
    f1_train = []
    f1_test = []

    for c in np.multiply([0.001, 0.01,0.1,1,10,100],100):
        clf= sklearn.svm.SVC(C=c,kernel='rbf').fit(x_train,y_train)
        y_pred_train=clf.predict(x_train)
        y_pred_test=clf.predict(x_test)
        score_train.append(metrics.accuracy_score(y_pred_train, y_train))
        score_test.append(metrics.accuracy_score(y_pred_test, y_test))
        f1_train.append(metrics.f1_score(y_pred_train,y_train))
        f1_test.append(metrics.f1_score(y_pred_test,y_test))
        #print("Report per fattore di penalizzazione:",c)                    #fighissimo ma stampa un sacco di cose
        #print(sklearn.metrics.classification_report(y_pred_test, y_test))   #dopo commentalo
    """
    print("Radial basis function kernel")
    print("score_train",score_train)
    print("f1_train",f1_train)
    print("score_test",score_test)
    print("f1_test", f1_test)
    """

    #LINEAR KERNEL
    score_train = []
    score_test = []
    f1_train = []
    f1_test = []
    for c in [0.001, 0.01,0.1,1,10,100]:
        clf= sklearn.svm.SVC(C=c,kernel='linear').fit(x_train,y_train)
        y_pred_train=clf.predict(x_train)
        y_pred_test=clf.predict(x_test)
        score_train.append(metrics.accuracy_score(y_pred_train, y_train))
        score_test.append(metrics.accuracy_score(y_pred_test, y_test))
        f1_train.append(metrics.f1_score(y_pred_train,y_train))
        f1_test.append(metrics.f1_score(y_pred_test,y_test))
        #print("Report per fattore di penalizzazione:",c)                    #fighissimo ma stampa un sacco di cose
        #print(sklearn.metrics.classification_report(y_pred_test, y_test))   #dopo commentalo
    """
    print("Linear kernel")
    print("score_train",score_train)
    print("f1_train",f1_train)
    print("score_test",score_test)
    print("f1_test",f1_test)
    """
    #il lineare probabilmente va meglio perchè abbiamo molti attributi 0-1 che sono linearmente separabili
    #il migliore con il kernel lineare sembra essere C=0.01

    #plot training and test
    xrange = [0.001, 0.01,0.1,1,10,100]
    f1error_train = np.ones(len(xrange)) - f1_train
    f1error_test = np.ones(len(xrange)) - f1_test
    error_train = np.ones(len(xrange)) - score_train
    error_test = np.ones(len(xrange)) - score_test
    plt.plot(xrange, error_train, label="train score accuracy", color='navy')
    plt.plot(xrange, error_test, label="test score accuracy", color='green')
    plt.plot(xrange, f1error_train, label="train F1 accuracy", color='royalblue')
    plt.plot(xrange, f1error_test, label="test F1 accuracy", color='lightgreen')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('SVM train and test F1 and score accuracy for different C')
    plt.legend()
    plt.show()

    #calcolo confusion matrix e cv accuracy
    clf=sklearn.svm.SVC(C=0.01,kernel='linear').fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
    matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    print("Confusion matrix for c=0.01 normalized by true categories (rows):")
    print(matrix)
    print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)
    print("Report del test set per fattore di penalizzazione=", 0.01)
    print(sklearn.metrics.classification_report(y_pred, y_test))

    #10-fold cross validation accuracy for k=5 is: 0.9259855769230769 -> sembra ottimo!

    #ora provo a usare i pesi per le classi, faccio un ciclo for con i vari pesi
    # LINEAR KERNEL
    score_train = []
    score_test = []
    f1_train = []
    f1_test = []

    #class weight funziona come = {valore label : peso da assegnare}
    # non è spiegato bene, penso vada ad operare sulla loss function e penalizza di più classifichi male un punto della
    #label selezionata
    for weight in [1,1.25,1.5,1.75,2,5,10]:
        clf = sklearn.svm.SVC(C=0.01, kernel='linear',class_weight={0:weight}).fit(x_train, y_train)
        y_pred_train = clf.predict(x_train)
        y_pred_test = clf.predict(x_test)
        score_train.append(metrics.accuracy_score(y_pred_train, y_train))
        score_test.append(metrics.accuracy_score(y_pred_test, y_test))
        f1_train.append(metrics.f1_score(y_pred_train, y_train))
        f1_test.append(metrics.f1_score(y_pred_test, y_test))
        # print("Report per fattore di penalizzazione:",c)                    #fighissimo ma stampa un sacco di cose
        # print(sklearn.metrics.classification_report(y_pred_test, y_test))   #dopo commentalo

    print("Linear kernel")
    print("score_train",score_train)
    print("f1_train",f1_train)
    print("score_test",score_test)
    print("f1_test",f1_test)

    clf=sklearn.svm.SVC(C=0.01,kernel='linear',class_weight={0:1.25}).fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
    matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
    print("Confusion matrix for c=0.01 normalized by true categories (rows):")
    print(matrix)
    print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)
    print("Report del test set per fattore di penalizzazione=", 0.01)
    print(sklearn.metrics.classification_report(y_pred, y_test))
    #aumenta la precision ma diminuisce il recall e in generale sembra che le prestazioni siano peggiori
    #il numero di previsioni è maggiore, spiegare bene nel report come funziona

    #sarebbe interessante vederlo con kernel rbf, magari quello funziona meglio

    #IDEA-> mettere variabile di controllo su tutte le funzioni, se voglio solo l'accuracy mi stampa solo quella,
    # in questo modo posso vedere tanti metodi velocemente senza avere troppe informazioni inutili e magari utile per report