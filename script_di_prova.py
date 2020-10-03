import pandas as pd

train_por = pd.read_csv('./student-por.csv', sep=';')
train_mat = pd.read_csv('./student-mat.csv', sep=';')

train = pd.concat([train_por, train_mat], axis=0)

del train['G1']
del train['G2']
del train['G3']
del train['absences']
del train['studytime']
del train['failures']

print(train.shape)
train = train.drop_duplicates(subset=None, keep='first', inplace=False)
print("After remove duplicates: " + str(train.shape))