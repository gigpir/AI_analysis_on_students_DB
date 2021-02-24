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
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
#from six import StringIO
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus

def kNN(x, y, onlynum=False, search=False, cv=True, k_cv=5, onlycv=False, smote=False, select='all',return_clf=False):
    if not smote:
        print('kNN classifier')
    else:
        print('kNN classifier with SMOTE')
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
        x=master.select_numerical(x, select=select)
    x_train, x_test, y_train, y_test = master.split(x, y, scaled=True)
    if smote:
        x_train,y_train=master.SMOTE(x_train, y_train)

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
        cv_accuracy=master.cross_val(neigh, x, y)
        if smote:
            cv_accuracy=master.cv_SMOTE(neigh,x,y)
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
    if return_clf:
        return neigh

def LDA(x,y, onlycv=False, testacc=False, smote=False):
    if not smote:
        print('LDA classifier')
    else:
        print('LDA classifier with SMOTE')
    #testacc=True -> stampa la test (validaton) accuracy (se onlycv=False)
    #onlycv=True -> stampa solo la cross validation accuracy


    x_train, x_test, y_train, y_test = master.split(x,y)
    if smote:
        x_train,y_train= master.SMOTE(x_train,y_train)

    clf = LinearDiscriminantAnalysis() #ho provato ad usare lo shrinkage con un solver diverso -> forma di regolarizzazione
    clf.fit(x_train, y_train)           #ma l'accuracy è più o meno la stessa e così non dobbiamo spiegarla nel report :D
    y_pred_test = clf.predict(x_test)
    score_test=metrics.accuracy_score(y_pred_test, y_test)

    if testacc==True and onlycv==False:
        print('Test score accuracy is:', score_test)
        print()

    cv_accuracy = master.cross_val(clf, x, y)
    if smote:
        cv_accuracy= master.cv_SMOTE(clf,x,y)
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


def logistic_regression(x,y, C_cv=1, search=False, cv=True, onlycv=False, smote=False, return_clf = False):
    if not smote:
        print('Logistic regression classifier')
    else:
        print('Logistic regression classifier with SMOTE')
    #nella logistic_regression non c'è bisogno di fare nessuna operazione di standardizzazione

    x_train, x_test, y_train, y_test = master.split(x,y)
    if smote:
        x_train,y_train= master.SMOTE(x_train,y_train)

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
        plt.ylabel('Score error')
        plt.title('Logistic Regression train and test score error for different values of C')
        plt.legend()
        plt.show()

    #Cross Validation
    if cv:
        clf=sklearn.linear_model.LogisticRegression(C=C_cv,max_iter=1000)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=master.cross_val(clf, x, y)
        if smote:
            cv_accuracy=master.cv_SMOTE(clf,x.y)
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

    if return_clf:
        return clf

def SVM(x,y, search=False, cv=True, C_cv=0.01, mode_cv='linear', onlycv=False, smote=False):
    if not smote:
        print('SVM classifier')
    else:
        print('SVM classifier with SMOTE')

    x_train, x_test, y_train, y_test = master.split(x,y)

    if smote:
        x_train, y_train = master.SMOTE(x_train, y_train)
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
        plt.ylabel('Score error')
        plt.title('SVM with linear kernel train and test score accuracy for different C')
        plt.legend()
        plt.show()

    #calcolo confusion matrix e cv accuracy
    if cv:
        clf=sklearn.svm.SVC(C=C_cv,kernel=mode_cv).fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        cv_accuracy=master.cross_val(clf, x, y)
        if smote:
            cv_accuracy = master.cv_SMOTE(clf, x, y)
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
        for weight in [1, 1.2, 1.4, 1.6, 1.8, 2]:
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
        cv_accuracy=master.cross_val(clf, x, y)
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


def decisionTree(x, y, feature_names, onlycv=False, smote=False):
    #per fare un print leggibile devo passargli il nome degli attributi
    #DecisionTreeClassifier non accetta attributi categorici

    if not smote:
        print('Decision Tree Classifier')
    else:
        print('Decision Tree Classifier with SMOTE')

    x_train, x_test, y_train, y_test = master.split(x, y)

    if smote:
        x_train,y_train=master.SMOTE(x_train,y_train)

    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_samples_leaf=40) #cv_acc= 0.91055
    #clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0) #cv_acc=0.89514

    clf = clf.fit(x_train, y_train)

    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    # Making the Confusion Matrix

    cm = confusion_matrix(y_test, y_pred_test, normalize="true")

    score_train = metrics.accuracy_score(y_pred_train, y_train)
    score_test = metrics.accuracy_score(y_pred_test, y_test)

    if not onlycv:
        print('The features sorting by descending importance are:')
        importance= clf.feature_importances_
        dictionary=dict(zip(feature_names,importance))
        dictionary=sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        for i in dictionary:
            if i[1]>0:
                print(i)
        #print(dictionary) #da associare alle colonne
        print()

        #r=export_text(clf, feature_names=feature_names)
        #print(r)

        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(clf,
                           feature_names=feature_names,
                           class_names=['fail','pass'],
                           filled=True)
        fig.savefig("./decision_tree/decistion_tree.png")
        plt.close('all')

        print("Confusion matrix normalized by true categories (rows):")
        print(cm)
        print()
        print("Report del test set")
        print(sklearn.metrics.classification_report(y_pred_test, y_test))
        #print('Train Score: '+str(score_train))
        #print('Test Score: '+str(score_test))
        #print()

    cv_accuracy = master.cross_val(clf, x, y)
    if smote:
        cv_accuracy=master.cv_SMOTE(clf,x,y)
    print('Cross validation accuracy:',cv_accuracy)


def randomForest(x,y, feature_names, search=False, cv=True, onlycv=False, crit_cv='gini', n_cv=100, smote=False):
    # Random Forest Classifier
    if not smote:
        print('Random Forest Classifier')
    else:
        print('Random Forest Classifier with SMOTE')

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    if smote:
        x_train, y_train = master.SMOTE(x_train, y_train)

    # Fitting the classifier into the Training set
    # Grid search for criterion and n_estimators
    print_e = ['Test score with entropy criterion']
    print_g = ['Test score with gini criterion']
    if search:
        for crit in ['entropy', 'gini']:
            for n in [10,50,100,200,500]:
                classifier = RandomForestClassifier(n_estimators=n, criterion=crit, random_state=0,min_samples_leaf=40)
                classifier.fit(x_train, y_train)
                y_pred_train = classifier.predict(x_train)
                y_pred_test = classifier.predict(x_test)
                score_train = metrics.accuracy_score(y_pred_train, y_train)
                score_test = metrics.accuracy_score(y_pred_test, y_test)
                if crit=='entropy':

                    print_e.append(score_test)
                else:

                    print_g.append(score_test)
                #print('Train Scorefor {} estimators and {} criterion:'.format(n,crit)+ str(score_train)) #per n>10 è sempre 1
                #print('Test Score for {} estimators and {} criterion:'.format(n,crit) + str(score_test))
        # Predicting the test set results
        header=[]
        for i in [10,50,100,200,500]:
            header.append("n={}".format(i))
        print(tabulate([print_e, print_g], headers=header))
        print()

    if cv:
        classifier= RandomForestClassifier(n_estimators=n_cv, criterion=crit_cv, random_state=0)
        classifier.fit(x_train, y_train)
        y_pred_test = classifier.predict(x_test)
        cv_accuracy = master.cross_val(classifier, x, y)
        if smote:
            cv_accuracy=master.cv_SMOTE(classifier,x,y)
        print("10-fold cross validation accuracy for {} estimators and {} criterion is:".format(n_cv,crit_cv), cv_accuracy)
        print()

        if not onlycv:
            # Making the Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test, normalize='true')
            print("Confusion matrix for {} estimators and {} criterion is:".format(n_cv,crit_cv))
            print(cm)
            print()
            print("Report del test set per {} estimators and {} criterion is:".format(n_cv,crit_cv))
            print(sklearn.metrics.classification_report(y_pred_test, y_test))
            print()
            print('The features sorting by descending importance are:')
            importance = classifier.feature_importances_
            dictionary = dict(zip(feature_names, importance))
            dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
            for i in dictionary:
                if i[1] > 0:
                    print(i)
            # print(dictionary) #da associare alle colonne
            print()


def ROC_curve_best_models(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.25,
                                                        random_state=1234, stratify=y)

    classifiers = [sklearn.linear_model.LogisticRegression(random_state=1234,C=0.1,max_iter=1000),
                   KNeighborsClassifier(n_neighbors=5),
                   LinearDiscriminantAnalysis(),
                   sklearn.svm.SVC(C=0.01, kernel='linear',probability=True),
                   sklearn.tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_samples_leaf=40),
                   RandomForestClassifier(random_state=1234, criterion='gini', n_estimators=10)]

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    for cls in classifiers:
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::, 1]

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, yproba)
        auc = sklearn.metrics.roc_auc_score(y_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.savefig('./ROC.png', dpi=400)
