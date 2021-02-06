import pandas as pd

'''train_por = pd.read_csv('./student-por.csv', sep=';')
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
print("After remove duplicates: " + str(train.shape))'''

import master

df = master.init()

y = df[['G1','G2','G3']]
X = df.drop(['G1','G2','G3'],axis=1)

df['pass_fail'] = df.apply(lambda row: master.label_pass(row), axis=1)


replace_binary_attributes_map = {'school': {'GP': 0, 'MS': 1},
                                     # school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
                                     'sex': {'F': 0, 'M': 1},
                                     # sex - student's sex (binary: 'F' - female or 'M' - male)
                                     'address': {'U': 0, 'R': 1},
                                     # address - student's home address type (binary: 'U' - urban or 'R' - rural)
                                     'famsize': {'LE3': 0, 'GT3': 1},
                                     # famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
                                     'Pstatus': {'T': 0, 'A': 1},
                                     # Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
                                     'schoolsup': {'yes': 1, 'no': 0},
                                     # schoolsup - extra educational support (binary: yes or no)
                                     'famsup': {'yes': 1, 'no': 0},
                                     # famsup - family educational support (binary: yes or no)
                                     'paid': {'yes': 1, 'no': 0},
                                     # paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
                                     'activities': {'yes': 1, 'no': 0},
                                     # activities - extra-curricular activities (binary: yes or no)
                                     'nursery': {'yes': 1, 'no': 0},
                                     # nursery - attended nursery school (binary: yes or no)
                                     'higher': {'yes': 1, 'no': 0},
                                     # higher - wants to take higher education (binary: yes or no)
                                     'internet': {'yes': 1, 'no': 0},
                                     # internet - Internet access at home (binary: yes or no)
                                     'romantic': {'yes': 1, 'no': 0},
                                     # romantic - with a romantic relationship (binary: yes or no)
                                     'pass_fail': {'pass': 1, 'fail': 0},  # “pass” if G3>=10 else “fail”
                                     }
df_ = df.copy()

df_.replace(replace_binary_attributes_map, inplace=True)

