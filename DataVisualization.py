import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

verbose = False


def performDataVis():
    train_por = pd.read_csv('./student-por.csv', sep=';')
    train_mat = pd.read_csv('./student-mat.csv', sep=';')

    if verbose:  # Check for dimensions, null values, duplicates
        print(train_por.head())
        print(train_mat.head())
        print()
        print("Portuguese Table size: " + str(train_por.shape))
        print("After drop null tuples: " + str(train_por.dropna().shape))  # no null values in train_por
        subset = train_por.columns
        train_por = train_por.drop_duplicates(subset=None, keep='first', inplace=False)
        print("After remove duplicates: " + str(train_por.shape))  # no duplicates in train_por

        print()
        print("Math Table size: " + str(train_mat.shape))
        print("After drop null tuples: " + str(train_mat.dropna().shape))  # no null values in train_mat
        subset = train_mat.columns
        train_mat = train_mat.drop_duplicates(subset=None, keep='first', inplace=False)
        print("After remove duplicates: " + str(train_mat.shape))  # no duplicates in train_mat

    # Join the 2 tables
    train_por['Subject'] = 'Portuguese'
    train_mat['Subject'] = 'Maths'

    train = pd.concat([train_por, train_mat], axis=0)

    if verbose:
        print('Concatenated Tables:')
        print(train.head())

    # save to csv
    train.to_csv('./student.csv', index=False, sep=';')

    data = pd.read_csv('./student.csv', sep=';')
    def correlation(df):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(20, 15))
        colormap = sns.diverging_palette(180, 350, as_cmap=True)
        sns.heatmap(corr, cmap=colormap, annot=True, fmt='.2f')
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig('./EDA/correlation.png', bbox_inches='tight')
    correlation(data)

    if verbose:
        print(data.columns)

    #School Distribution plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='school', data = data, order=['GP', 'MS'])
    ax = ax.set(ylabel="Count", xlabel="school")
    figure.grid(False)
    plt.title('School Distribution')
    plt.savefig('./EDA/school.png', bbox_inches='tight')

    #Gender Distribution plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='sex', data=data, order=['M', 'F'])
    ax = ax.set(ylabel="Count", xlabel="gender")
    figure.grid(False)
    plt.title('Gender Distribution')
    plt.savefig('./EDA/gender.png', bbox_inches='tight')

    #Age plot
    if verbose:
        print('Age:\nMin: ' + str(data['age'].min())+
              'Max: ' + str(data['age'].max()))

    age_range = np.arange(data['age'].min(), data['age'].max())

    f, ax = plt.subplots()
    figure = sns.countplot(x='age', data=data, order=age_range)
    ax = ax.set(ylabel="Count", xlabel="age")
    figure.grid(False)
    plt.title('Age Distribution')
    plt.savefig('./EDA/age.png', bbox_inches='tight')