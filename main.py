import master
import analysis
import DataVisualization

def main():

    DataVisualization.performDataVis()


    #df = master.init()
    #print(df)

    #X, y = master.preproc(df)

    #analysis.preliminaryStat(X, df['binary'])

    #analysis.heatMap(X,y)
    #master.kNN(X,y)

    #X_num= master.select_numerical(X)

    #master.kNN(X_num,y)

    #metto prima lda che fa un po' schifo
    #master.LDA(X,y)

    #master.logistic_regression(X, y)

    #master.SVD(X,y)


if __name__ == "__main__":
    main()