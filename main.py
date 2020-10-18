import master
import analysis
import DataVisualization
import bclassification

def main():

    #DataVisualization.performDataVis()

    #master.studydatasets()

    df= master.init('por')
    X,y, feature_names = master.preproc(df, select='novotes')

    #df = master.init()
    #print(df)

    #X, y, feature_names = master.preproc(df)

    #bclassification.decisionTree(X,y, feature_names)
    #bclassification.decisionTree(X, y, feature_names, onlycv=True)
    #bclassification.decisionTree(X,y, feature_names, onlycv=True, smote=True)

    #bclassification.randomForest(X, y, feature_names, search=True)

    #bclassification.SVM(X,y, onlycv=True)

    #analysis.preliminaryStat(X, df['binary'])

    #analysis.heatMap(X,y)

    bclassification.kNN(X, y, search=True, cv=False)

    #bclassification.kNN(X,y,onlynum=True, search=True, cv=False)

    #bclassification.kNN(X, y)

    #bclassification.kNN(X, y, onlycv=True) #0.899

    #bclassification.kNN(X,y,onlynum=True, onlycv=True) #0.9075

    #metto prima lda che fa un po' schifo
    #bclassification.LDA(X,y) #0.8828

    #bclassification.logistic_regression(X, y, onlycv=True) #0.9105

    #bclassification.SVM(X,y,search=True,cv=False)
    #bclassification.SVM(X, y)

    #bclassification.SVM_unbalanced(X,y,search=True, cv=False)
    #bclassification.SVM_unbalanced(X,y, onlycv=True)

    #master.PCA_study(X, feature_names=feature_names)

    #X_pca=master.PCA(X,38)

    #devo di nuovo fare tuning dei parametri perchè cambiano
    #per esempio in SVD
    #bclassification.SVM(X_pca, y, search=True, cv=False)
    #bclassification.SVM(X_pca, y, C_cv=0.1, mode_cv='linear', onlycv=True)
    #bclassification.kNN(X_pca,y,onlycv=True)
    #mi pare di vedere che ho sempre risultati un po' peggiori, anche con 38 componenti

    #bclassification.SVM(X,y,search=True, cv=False)
    #bclassification.SVM(X,y, smote=True,C_cv=100, mode_cv='rbf')


if __name__ == "__main__":
    main()