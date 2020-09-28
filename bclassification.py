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
import master
from tabulate import tabulate





def kNN(x, y, onlynum=False, search=False, cv=True, k_cv=5, onlycv=False):
    ####################
    # KNN
    ####################

    #onlynum=True -> seleziona solo gli attributi numerici, vedere master.onlynum(x)
    #search=True -> ricerca del parametro k ottimale e stampa grafico train e test error
    #cv=True -> stampa cross validation score, confusion matrix e classification report
    #k_cv -> se si vuole provare un parametro k diverso nella fase di cross validation
    #onlycv=True -> attenzione che cv deve essere True, stampa solo cv accuracy e non classreport e confmatr


    # KNN doesn't work well on datasets with many features and in particular with
    # sparse matrices (our case because we have a lot of categorical data)
    # https://medium.com/cracking-the-data-science-interview/k-nearest-neighbors-who-are-close-to-you-19df59b97e7d

    # divisione fra train e test set
    if onlynum:
        x=master.select_numerical(x)
    x_train, x_test, y_train, y_test = master.split(x, y, scaled=True)

    if search:
        score_train = []
        score_test = []
        print1=['Train score']
        print2=['Test score']

        # applico KNN con diversi parametri -> faccio da 1 a 19
        for i in range(1, 21, 2):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(x_train, y_train)

            # calcolo accuratezza del train e del test set -> studio overfitting
            y_pred_train = neigh.predict(x_train)
            y_pred_test = neigh.predict(x_test)

            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))
        header=[' ']
        for i in range(1,21,2):
            header.append("k={}".format(i))
        print(tabulate([print1,print2],headers=header))
        print()

        # Plot of train and test error for different values of K
        xrange = range(1, 21, 2)
        error_train = np.ones(len(xrange)) - score_train
        error_test = np.ones(len(xrange)) - score_test

        plt.plot(xrange, error_train, label="train error")
        plt.plot(xrange, error_test, label="test error")

        plt.xlabel('K')
        plt.ylabel('Score error')
        plt.title('KNN train and test score error for different values of K')
        plt.legend()
        plt.show()

    # ATTENZIONE-> non dovrei scegliere il parametro migliore sulla base del test set, dovrei avere
    # train-val-test. In questo caso il nostro test set può essere considerato come validation e considerare
    # come stimatore dell'accuracy la cross validation

    # rialleniamo KNN per visualizzare la matrice di confusione
    if cv==True:
        neigh = KNeighborsClassifier(n_neighbors=k_cv)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)
        cv_accuracy=np.mean(cross_val_score(neigh, x, y, cv=10))
        matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
        print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)
        print()
        if onlycv==False:
            print("Confusion matrix for k=5 normalized by true categories (rows):")
            print(matrix)
            print()
            print('Classification report:')
            print(sklearn.metrics.classification_report(y_pred, y_test))


def LDA(x,y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    x_train, x_test, y_train, y_test = master.split(x,y)

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