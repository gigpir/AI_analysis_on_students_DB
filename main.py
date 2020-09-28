import master
import analysis
import DataVisualization
import bclassification

def main():

    #DataVisualization.performDataVis()


    df = master.init()
    #print(df)

    X, y = master.preproc(df)

    #analysis.preliminaryStat(X, df['binary'])

    #analysis.heatMap(X,y)

    bclassification.kNN(X, y, search=True, cv=False)

    #bclassification.kNN(X,y,onlynum=True, search=True, cv=False)

    #bclassification.kNN(X, y)

    #bclassification.kNN(X, y, onlycv=True)

    #bclassification.kNN(X,y,onlynum=True)

    #metto prima lda che fa un po' schifo
    #master.LDA(X,y)

    #master.logistic_regression(X, y)

    #master.SVD(X,y)


if __name__ == "__main__":
    main()