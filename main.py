import master


def main():
    df = master.init()
    print(df)

    X, y = master.preproc(df)

    master.kNN(X,y)




if __name__ == "__main__":
    main()