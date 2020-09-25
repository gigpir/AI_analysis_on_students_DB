import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import matplotlib.pyplot as plt
from subprocess import check_output


def preliminaryStat(X, y):
    map = {0: 'Failed', 1: 'Passed'}
    y.replace(map, inplace=True)
    #y.columns = ['Final Result']
    yasd=y
    ax = sns.countplot(yasd, x='Results')
    plt.show()
    P, F = y.value_counts()
    print('Number of Pass: ', P)
    print('Number of Fail : ', F)
