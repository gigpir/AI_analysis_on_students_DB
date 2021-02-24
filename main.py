import master
import analysis
import DataVisualization
import bclassification
import time

def main():

    #DataVisualization.performDataVis()

    #reading the dataset
    df= master.init('por')

    #preprocessing -> feature transformation
    X,y, feature_names = master.preproc(df, select='novotes')
    X2, y2, feature_names2 = master.preproc(df, select='G1')
    X3, y3, feature_names3= master.preproc(df,select='all')

    # applying and testing PCA
    """
    master.PCA_study(X,feature_names) #43 components
    #27 componenti -> 0.85
    #30 componenti -> 0.90
    #39 componenti -> 0.99

    X_pca_30 = master.PCA(X, components=30)

    t0=time.time()
    bclassification.kNN(X_pca_30, y, search=False, cv=True, onlycv=True) #0.82
    time_30= time.time()-t0
    print("Time elapsed: ", time_30) #0.16

    t1=time.time()
    bclassification.kNN(X, y, search=False, cv=True, onlycv=True) #0.83
    time_all=time.time()-t1
    print("Time elapsed: ", time_all) #0.22

    t2=time.time()
    bclassification.SVM(X_pca_30, y, search=False, cv=True, onlycv=True) #0.84
    time_30= time.time()-t2
    print("Time elapsed: ", time_30) #0.20

    t3=time.time()
    bclassification.SVM(X, y, search=False, cv=True, onlycv=True) #0.84
    time_all=time.time()-t3
    print("Time elapsed: ", time_all) #0.28
     """

    # Binary classification -> hyperparameter tuning and cross validation
    #bclassification.kNN(X, y, search=True, cv=False, select='novotes')
    #bclassification.kNN(X,y,search=False,cv=True, select='novotes')
    #bclassification.kNN(X,y,search=False,cv=True, select='novotes', smote=True)
    #bclassification.kNN(X,y,search=False,cv=True, select='novotes', onlynum=True)

    #bclassification.logistic_regression(X,y,search=True, cv=False)
    #bclassification.logistic_regression(X,y,search=False, cv=True)
    #bclassification.logistic_regression(X,y,search=False, cv=True,smote=True)

    #bclassification.LDA(X,y)
    #bclassification.LDA(X,y,smote=True)

    #bclassification.SVM(X,y,search=True, cv=False)
    #bclassification.SVM(X,y)
    #bclassification.SVM(X,y,smote=True)
    #bclassification.SVM(X,y,mode_cv='rbf',C_cv=10)
    #bclassification.SVM(X,y,mode_cv='rbf',C_cv=100, smote=True)

    #bclassification.SVM_unbalanced(X,y,search=True,cv=False)
    #bclassification.SVM_unbalanced(X,y)

    #bclassification.decisionTree(X,y,feature_names)
    #bclassification.decisionTree(X,y,feature_names,smote=True)

    #bclassification.randomForest(X,y,feature_names,search=True,cv=False)
    #bclassification.randomForest(X,y,feature_names)
    #bclassification.randomForest(X, y, feature_names, smote=True)

    # Perform binary classification on different configurations of the dataset:
    # X -> without G1 and G2
    # X2 -> with G1
    # X3 -> with G1 and G2
    """
    bclassification.kNN(X,y,onlycv=True)
    bclassification.kNN(X2, y2, onlycv=True)
    bclassification.kNN(X3, y3, onlycv=True)
    bclassification.LDA(X,y,onlycv=True)
    bclassification.LDA(X2, y2, onlycv=True)
    bclassification.LDA(X3, y3, onlycv=True)
    bclassification.logistic_regression(X,y,onlycv=True)
    bclassification.logistic_regression(X2, y2, onlycv=True)
    bclassification.logistic_regression(X3, y3, onlycv=True)
    bclassification.SVM(X,y,onlycv=True)
    bclassification.SVM(X2, y2, onlycv=True)
    bclassification.SVM(X3, y3, onlycv=True)
    bclassification.SVM_unbalanced(X,y,onlycv=True)
    bclassification.SVM_unbalanced(X2, y2, onlycv=True)
    bclassification.SVM_unbalanced(X3, y3, onlycv=True)
    bclassification.decisionTree(X,y,feature_names, onlycv=True)
    bclassification.decisionTree(X2, y2, feature_names2, onlycv=True)
    bclassification.decisionTree(X3, y3, feature_names3, onlycv=True)
    bclassification.randomForest(X,y,feature_names,onlycv=True)
    bclassification.randomForest(X2, y2, feature_names2, onlycv=True)
    bclassification.randomForest(X3, y3, feature_names3, onlycv=True)
    """

    #visualize ROC curve
    #bclassification.ROC_curve_best_models(X,y)

if __name__ == "__main__":
    main()