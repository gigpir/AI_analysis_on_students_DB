import pandas as pd
import numpy as np
import os

####################
#IMPORT DATASET
####################
os.listdir(os.getcwd())
df= pd.read_csv("/home/user/student-por-new.csv",sep=";")
#print(df.head())
#print(df.columns)
#df.info()

####################
#PREPROCESSING
####################

#Create a map to perform mapping of binary attributes ONLY

replace_binary_attributes_map = {'school': {'GP': 0, 'MS': 1}, #school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
               'sex'    : {'F' : 0, 'M' : 1},     #sex - student's sex (binary: 'F' - female or 'M' - male)
               'address': {'U' : 0, 'R' : 1},     #address - student's home address type (binary: 'U' - urban or 'R' - rural)
               'famsize': {'LE3' : 0, 'GT3' : 1}, #famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
               'Pstatus': {'T' : 0, 'A' : 1},     #Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
               'schoolsup' : {'yes' : 1, 'no' : 0}, #schoolsup - extra educational support (binary: yes or no)
               'famsup' : {'yes' : 1, 'no' : 0}, #famsup - family educational support (binary: yes or no)
               'paid' : {'yes' : 1, 'no' : 0}, #paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
               'activities' : {'yes' : 1, 'no' : 0}, #activities - extra-curricular activities (binary: yes or no)
               'nursery' : {'yes' : 1, 'no' : 0}, #nursery - attended nursery school (binary: yes or no)
               'higher' : {'yes' : 1, 'no' : 0}, #higher - wants to take higher education (binary: yes or no)
               'internet' : {'yes' : 1, 'no' : 0}, #internet - Internet access at home (binary: yes or no)
               'romantic' : {'yes' : 1, 'no' : 0}, #romantic - with a romantic relationship (binary: yes or no)
               'binary' : {'pass' : 1, 'fail' : 0}, #“pass” if G3>=10 else “fail”
               }

df_2 = df.copy()

df_2.replace(replace_binary_attributes_map, inplace=True)

pd.set_option('display.max_columns', None)
#print(df.head())
#print(df_2.head())
#print(df_2['Mjob'].value_counts())


#Encode categorical data into new binary labels
df_2=pd.get_dummies(df_2, columns=['Mjob'])
df_2=pd.get_dummies(df_2, columns=['Fjob'])
df_2=pd.get_dummies(df_2, columns=['reason'])
df_2=pd.get_dummies(df_2, columns=['guardian'])

#print(df_2.head())

#print(df_2.columns)

#Re order the table
db=df_2[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','Mjob_at_home', 'Mjob_health', 'Mjob_other',
       'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
       'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
       'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
       'guardian_mother', 'guardian_other', 'G1', 'G2',
       'G3', 'binary', '5-class']]

#print(db.head())

x=db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','Mjob_at_home', 'Mjob_health', 'Mjob_other',
       'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
       'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
       'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
       'guardian_mother', 'guardian_other', 'G1', 'G2']]
x_novotes=db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu']]
y=db[['binary']]
#print(x)
#print(y)


####################
#KNN
####################

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# KNN doesn't work well on datasets with many features and in particular with 
# sparse matrices (our case because we have a lot of categorical data)
# https://medium.com/cracking-the-data-science-interview/k-nearest-neighbors-who-are-close-to-you-19df59b97e7d

# Potrebbe essere una buona idea cercare di selezionare solo gli attributi numerici e togliere gli attributi categorici

#trasformo il mio dataset in un array numpy
x_new=np.array(x)
y=np.array(y)
y=np.ravel(y)

#preprocessing -> standardizzazione
x_scaled=preprocessing.scale(x_new)

#print(x_scaled)
#print(y.shape)
#for i in range(0,5):
#    neigh = KNeighborsClassifier(n_neighbors=i)
#   scores[i] = cross_val_score(neigh, x_scaled, y, cv=10)
#print(scores)

#divisione fra train e test set
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.5, random_state=123)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)

score_train=[]
score_test=[]
score_cross=[]
# applico KNN con diversi parametri -> faccio da 1 a 11 a passi di 2 perchè deve essere dispari e dopo
# una certa soglia si stabilizza e non è più interessante
# calcolo la 10-fold cross validation accuracy 
for i in range(1,21,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(x_train,y_train)

    score_cross.append(np.mean(cross_val_score(neigh,x_scaled,y,cv=10)))

    #calcolo accuratezza del train e del test set -> studio overfitting
    y_pred_train=neigh.predict(x_train)
    y_pred_test=neigh.predict(x_test)

    score_train.append(metrics.accuracy_score(y_pred_train, y_train))
    score_test.append(metrics.accuracy_score(y_pred_test,y_test))

print(score_train)
print(score_test)
print(score_cross)


# Plot of train and test error for different values of K
xrange = range(1,21,2)
error_train= np.ones(len(xrange)) - score_train
error_test= np.ones(len(xrange)) - score_test
#error_cross= np.ones(len(xrange)) - score_cross
plt.plot(xrange, error_train, label = "train accuracy")
plt.plot(xrange, error_test , label = "test accuracy")
#plt.plot(xrange, error_cross, label = 'cross val accuracy') if I want also cv error
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('KNN train, test and CV error for different values of K')
plt.legend()
plt.show()

#la scelta migliore con train-test è k=5 -> test_accuracy=0.86769 (val)

#ATTENZIONE-> non dovrei scegliere il parametro migliore sulla base del test set, dovrei avere
# train-val-test. In questo caso il nostro test set può essere considerato come validation e considerare
# come stimatore dell'accuracy la cross validation

#rialleniamo KNN per visualizzare la matrice di confusione
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(x_train,y_train)
y_pred=neigh.predict(x_test)
matrix=metrics.confusion_matrix(y_test,y_pred)
print(matrix) #forse è più significativo se metto le accuratezze

# Bisognerebbe provare ad usare solo le variabili numeriche

x=db[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences','Mjob_at_home', 'Mjob_health', 'Mjob_other',
       'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health',
       'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course',
       'reason_home', 'reason_other', 'reason_reputation', 'guardian_father',
       'guardian_mother', 'guardian_other', 'G1', 'G2']]
