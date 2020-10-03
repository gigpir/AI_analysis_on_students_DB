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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def kNN(x, y, onlynum=False, search=False, cv=True, k_cv=5, onlycv=False):
    print('kNN classifier')
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
    if cv:
        neigh = KNeighborsClassifier(n_neighbors=k_cv)
        neigh.fit(x_train, y_train)
        y_pred = neigh.predict(x_test)
        cv_accuracy=np.mean(cross_val_score(neigh, x, y, cv=10))
        matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
        print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)
        print()
        if not onlycv:
            print("Confusion matrix for k=5 normalized by true categories (rows):")
            print(matrix)
            print()
            print('Classification report:')
            print(sklearn.metrics.classification_report(y_pred, y_test))
            print()


def LDA(x,y, onlycv=False, testacc=False):
    print('LDA classifier')
    #testacc=True -> stampa la test (validaton) accuracy (se onlycv=False)
    #onlycv=True -> stampa solo la cross validation accuracy


    x_train, x_test, y_train, y_test = master.split(x,y)

    clf = LinearDiscriminantAnalysis() #ho provato ad usare lo shrinkage con un solver diverso -> forma di regolarizzazione
    clf.fit(x_train, y_train)           #ma l'accuracy è più o meno la stessa e così non dobbiamo spiegarla nel report :D
    y_pred_test = clf.predict(x_test)
    score_test=metrics.accuracy_score(y_pred_test, y_test)

    if testacc==True and onlycv==False:
        print('Test score accuracy is:', score_test)
        print()

    cv_accuracy = np.mean(cross_val_score(clf, x, y, cv=10))
    print("10-fold cross validation accuracy for k=5 is:", cv_accuracy)
    print()

    if onlycv==False:
        matrix = metrics.confusion_matrix(y_test, y_pred_test, normalize="true")
        print("Confusion matrix for k=5 normalized by true categories (rows):")
        print(matrix)
        print()
        print('Classification report:')
        print(sklearn.metrics.classification_report(y_pred_test, y_test))
        print()


def logistic_regression(x,y, C_cv=1, search=False, cv=True, onlycv=False):
    print('Logistic regression classifier')
    #nella logistic_regression non c'è bisogno di fare nessuna operazione di standardizzazione

    x_train, x_test, y_train, y_test = master.split(x,y)

    score_train = []
    score_test = []
    print1 = ['Train score']
    print2 = ['Test score']
    #search for better penalization parameter
    if search==True:
        for c in [0.001, 0.01,0.1,1,10,100]:
            clf=sklearn.linear_model.LogisticRegression(C=c,max_iter=1000).fit(x_train, y_train)
            y_pred_train=clf.predict(x_train)
            y_pred_test=clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

        header=[' ']
        for i in [0.001, 0.01,0.1,1,10,100]:
            header.append("C={}".format(i))
        print(tabulate([print1,print2],headers=header))
        print()

        #plot train and test score error
        xrange = [0.001, 0.01,0.1,1,10,100]
        error_train = np.ones(len(xrange)) - score_train
        error_test = np.ones(len(xrange)) - score_test
        plt.plot(xrange, error_train, label="train score error")
        plt.plot(xrange, error_test, label="test score error")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('Logistic Regression train and test score error for different values of C')
        plt.legend()
        plt.show()

    #Cross Validation
    if cv:
        clf=sklearn.linear_model.LogisticRegression(C=C_cv,max_iter=1000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
        print("10-fold cross validation accuracy for C={} is:".format(C_cv), cv_accuracy)
        print()

        if not onlycv:
            matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
            print("Confusion matrix for C={} normalized by true categories (rows):".format(C_cv))
            print(matrix)
            print()
            print("Report del test set per fattore di penalizzazione C=", C_cv)
            print(sklearn.metrics.classification_report(y_pred, y_test))
            print()


def SVM(x,y, search=False, cv=True, C_cv=0.01, mode_cv='linear', onlycv=False):
    print('SVM classifier')
    x_train, x_test, y_train, y_test = master.split(x,y)

    #normalizzazione?
    #non credo influisca, alla fine otterrei solo un iperpiano deformato

    if search:
        #RBF KERNEL
        score_train = []
        score_test = []
        print1 = ['Train score']
        print2 = ['Test score']

        for c in np.multiply([0.001, 0.01,0.1,1,10,100],100):
            clf= sklearn.svm.SVC(C=c,kernel='rbf').fit(x_train,y_train)
            y_pred_train=clf.predict(x_train)
            y_pred_test=clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

        header=[' ']
        for i in np.multiply([0.001, 0.01,0.1,1,10,100],100):
            header.append("C={}".format(i))
        print('RBF kernel')
        print(tabulate([print1,print2],headers=header))
        print()
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
        print1 = ['Train score']
        print2 = ['Test score']
        for c in [0.001, 0.01,0.1,1,10,100]:
            clf= sklearn.svm.SVC(C=c,kernel='linear').fit(x_train,y_train)
            y_pred_train=clf.predict(x_train)
            y_pred_test=clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

        header = [' ']
        for i in [0.001, 0.01,0.1,1,10,100]:
            header.append("C={}".format(i))
        print('Linear kernel')
        print(tabulate([print1, print2], headers=header))
        print()
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
        error_train = np.ones(len(xrange)) - score_train
        error_test = np.ones(len(xrange)) - score_test
        plt.plot(xrange, error_train, label="train score error")
        plt.plot(xrange, error_test, label="test score error")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('SVM with linear kernel train and test score accuracy for different C')
        plt.legend()
        plt.show()

    #calcolo confusion matrix e cv accuracy
    if cv:
        clf=sklearn.svm.SVC(C=C_cv,kernel=mode_cv).fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
        print("10-fold cross validation accuracy for C={} and {} kernel is:".format(C_cv,mode_cv), cv_accuracy)
        print()

        if not onlycv:
            matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
            print("Confusion matrix for C={} and {} kernel normalized by true categories (rows):".format(C_cv, mode_cv))
            print(matrix)
            print()
            print("Report del test set per fattore di penalizzazione C={} and {} kernel".format(C_cv, mode_cv))
            print(sklearn.metrics.classification_report(y_pred, y_test))
    #10-fold cross validation accuracy for k=5 is: 0.9259855769230769 -> sembra ottimo!


def SVM_SMOTE(x,y, search=False, cv=True, C_cv=100, mode_cv='rbf', onlycv=False):
    print('SVM classifier with SMOTE')
    x_train, x_test, y_train, y_test = master.split(x,y)
    x_train, y_train=master.SMOTE(x_train,y_train)

    #normalizzazione?
    #non credo influisca, alla fine otterrei solo un iperpiano deformato

    if search:
        #RBF KERNEL
        score_train = []
        score_test = []
        print1 = ['Train score']
        print2 = ['Test score']

        for c in np.multiply([0.001, 0.01,0.1,1,10,100],100):
            clf= sklearn.svm.SVC(C=c,kernel='rbf').fit(x_train,y_train)
            y_pred_train=clf.predict(x_train)
            y_pred_test=clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

        header=[' ']
        for i in np.multiply([0.001, 0.01,0.1,1,10,100],100):
            header.append("C={}".format(i))
        print('RBF kernel')
        print(tabulate([print1,print2],headers=header))
        print()

        #LINEAR KERNEL
        score_train = []
        score_test = []
        print1 = ['Train score']
        print2 = ['Test score']
        for c in [0.001, 0.01,0.1,1,10,100]:
            clf= sklearn.svm.SVC(C=c,kernel='linear').fit(x_train,y_train)
            y_pred_train=clf.predict(x_train)
            y_pred_test=clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

        header = [' ']
        for i in [0.001, 0.01,0.1,1,10,100]:
            header.append("C={}".format(i))
        print('Linear kernel')
        print(tabulate([print1, print2], headers=header))
        print()

        #il lineare probabilmente va meglio perchè abbiamo molti attributi 0-1 che sono linearmente separabili
        #il migliore con il kernel lineare sembra essere C=0.01

        #plot training and test
        xrange = [0.001, 0.01,0.1,1,10,100]
        error_train = np.ones(len(xrange)) - score_train
        error_test = np.ones(len(xrange)) - score_test
        plt.plot(xrange, error_train, label="train score error")
        plt.plot(xrange, error_test, label="test score error")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('SVM with linear kernel train and test score accuracy for different C')
        plt.legend()
        plt.show()

    #calcolo confusion matrix e cv accuracy
    if cv:
        clf=sklearn.svm.SVC(C=C_cv,kernel=mode_cv).fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=master.cv_SMOTE(clf,x,y)
        print("10-fold cross validation accuracy for C={} and {} kernel is:".format(C_cv,mode_cv), cv_accuracy)
        print()

        if not onlycv:
            matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
            print("Confusion matrix for C={} and {} kernel normalized by true categories (rows):".format(C_cv, mode_cv))
            print(matrix)
            print()
            print("Report del test set per fattore di penalizzazione C={} and {} kernel".format(C_cv, mode_cv))
            print(sklearn.metrics.classification_report(y_pred, y_test))
    #10-fold cross validation accuracy for k=5 is: 0.9259855769230769 -> sembra ottimo!


def SVM_unbalanced(x,y, search=False, cv=True, weight_cv=1.25, onlycv=False):
    print('SVM unbalanced classifier')
    # script per classi non bilanciate -> da provare e capire se si può applicare anche agli altri
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py

    x_train, x_test, y_train, y_test = master.split(x,y)

    # Uso il migliore risultato dell'SVD -> linear kernel e C=0.01

    if search:

        score_train = []
        score_test = []
        print1 = ['Train score']
        print2 = ['Test score']


        #class weight funziona come = {valore label : peso da assegnare}
        # non è spiegato bene, penso vada ad operare sulla loss function e penalizza di più classifichi male un punto della
        #label selezionata
        for weight in [1, 1.1, 1.2, 1.3, 1.4, 1.5]:
            clf = sklearn.svm.SVC(C=0.01, kernel='linear',class_weight={0:weight}).fit(x_train, y_train)
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            score_train.append(metrics.accuracy_score(y_pred_train, y_train))
            score_test.append(metrics.accuracy_score(y_pred_test, y_test))
            print1.append(metrics.accuracy_score(y_pred_train, y_train))
            print2.append(metrics.accuracy_score(y_pred_test, y_test))

            # print("Report per fattore di penalizzazione:",c)                    #fighissimo ma stampa un sacco di cose
            # print(sklearn.metrics.classification_report(y_pred_test, y_test))   #dopo commentalo

        header=[' ']
        for i in [1, 1.2, 1.4, 1.6, 1.8, 2]:
            header.append("weight={}".format(i))
        print(tabulate([print1,print2],headers=header))
        print()

    if cv:
        clf=sklearn.svm.SVC(C=0.01,kernel='linear',class_weight={0:weight_cv}).fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=np.mean(cross_val_score(clf, x, y, cv=10))
        print("10-fold cross validation accuracy for weight={} is:".format(weight_cv), cv_accuracy)
        print()

        if not onlycv:
            matrix = metrics.confusion_matrix(y_test, y_pred, normalize="true")
            print("Confusion matrix for weight={} normalized by true categories (rows):".format(weight_cv))
            print(matrix)
            print()

            print("Report del test set per weight=", weight_cv)
            print(sklearn.metrics.classification_report(y_pred, y_test))
            print()
