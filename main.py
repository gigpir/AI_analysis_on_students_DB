import master
import analysis
import DataVisualization
import bclassification

def main():

    DataVisualization.performDataVis()


    #df = master.init()
    #print(df)

    #X, y = master.preproc(df)


    #analysis.preliminaryStat(X, df['binary'])

    #analysis.heatMap(X,y)

    #bclassification.kNN(X, y, search=True, cv=False)

    #bclassification.kNN(X,y,onlynum=True, search=True, cv=False)

    #bclassification.kNN(X, y)

    #bclassification.kNN(X, y, onlycv=True)

    #bclassification.kNN(X,y,onlynum=True, onlycv=True)

    #metto prima lda che fa un po' schifo
    #bclassification.LDA(X,y)

    #bclassification.logistic_regression(X, y, onlycv=True)

    #bclassification.SVD(X,y,search=True,cv=False)
    #bclassification.SVM(X, y, onlycv=True)

    #bclassification.SVD_unbalanced(X,y,search=True, cv=False)
    #bclassification.SVD_unbalanced(X,y, onlycv=True)

    #master.PCA_study(X)

    #X_pca=master.PCA(X,38)

    #devo di nuovo fare tuning dei parametri perchè cambiano
    #per esempio in SVD
    #bclassification.SVM(X_pca, y, search=True, cv=False)
    #bclassification.SVM(X_pca, y, C_cv=0.1, mode_cv='linear', onlycv=True)
    #bclassification.kNN(X_pca,y,onlycv=True)
    #mi pare di vedere che ho sempre risultati un po' peggiori, anche con 38 componenti


if __name__ == "__main__":
    main()