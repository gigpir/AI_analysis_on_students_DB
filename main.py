import master
import analysis

def main():
    df = master.init()
    #print(df)

    X, y = master.preproc(df)

    analysis.preliminaryStat(X, df['binary'])

    #master.kNN(X,y)




if __name__ == "__main__":
    main()