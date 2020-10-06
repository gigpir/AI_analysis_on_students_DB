import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.colors as colors
from plotly.offline import plot

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

    #Address plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='address', data=data, order=['U', 'R'])
    ax = ax.set(ylabel="Count", xlabel="address")
    figure.grid(False)
    plt.title('Address Distribution')
    plt.savefig('./EDA/address.png', bbox_inches='tight')

    #Family plot
    if verbose:
        fam = data['famsize'].unique()
        #print('Family size values: '+ str(fam))

    f, ax = plt.subplots()
    figure = sns.countplot(x='famsize', data=data, order=['LE3', 'GT3'])
    ax = ax.set(ylabel="Count", xlabel="famsize")
    figure.grid(False)
    plt.title('Family Distribution')
    plt.savefig('./EDA/family.png', bbox_inches='tight')

    #Parent Status
    f, ax = plt.subplots()
    figure = sns.countplot(x='Pstatus', data=data, order=['A', 'T'])
    ax = ax.set(ylabel="Count", xlabel="status")
    figure.grid(False)
    plt.title('Parents status Distribution')
    plt.savefig('./EDA/Pstatus_plot.png', bbox_inches='tight')

    #parent edu
    # (numeric: 0 - none, 1 - primary education (4th grade),
    # 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
    f, ax = plt.subplots()
    figure = sns.countplot(x='Medu', data=data, order=[0, 1, 2, 3, 4])
    ax = ax.set(ylabel="Count", xlabel="Mother Education")
    figure.grid(False)
    plt.title('Mother Education Distribution')
    plt.savefig('./EDA/Mother_Education_plot.png', bbox_inches='tight')
    f, ax = plt.subplots()
    figure = sns.countplot(x='Fedu', data=data, order=[0, 1, 2, 3, 4])
    ax = ax.set(ylabel="Count", xlabel="Father Education")
    figure.grid(False)
    plt.title('Father Education Distribution')
    plt.savefig('./EDA/Father_Education_plot.png', bbox_inches='tight')

    # parent job
    # (nominal: 'teacher', 'health' care related,
    # civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    f, ax = plt.subplots()
    figure = sns.countplot(x='Mjob', data=data, order=['teacher', 'health', 'services', 'at_home', 'other'])
    ax = ax.set(ylabel="Count", xlabel="Mother Job")
    figure.grid(False)
    plt.title('Mother Job Distribution')
    plt.savefig('./EDA/mother_Job_plot.png', bbox_inches='tight')

    f, ax = plt.subplots()
    figure = sns.countplot(x='Fjob', data=data, order=['teacher', 'health', 'services', 'at_home', 'other'])
    ax = ax.set(ylabel="Count", xlabel="Father Job")
    figure.grid(False)
    plt.title('Father Job Distribution')
    plt.savefig('./EDA/father_Job_plot.png', bbox_inches='tight')

    #reason plot
    # (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
    f, ax = plt.subplots()
    figure = sns.countplot(x='reason', data=data, order=['home', 'reputation', 'course', 'other'])
    ax = ax.set(ylabel="Count", xlabel="reason to chose this school")
    figure.grid(False)
    plt.title('Reason Distribution')
    plt.savefig('./EDA/Reason_plot.png', bbox_inches='tight')

    #guardian plot
    # (nominal: 'mother', 'father' or 'other')
    f, ax = plt.subplots()
    figure = sns.countplot(x='guardian', data=data, order=['mother', 'father', 'other'])
    ax = ax.set(ylabel="Count", xlabel="Guardian")
    figure.grid(False)
    plt.title('Guardian Distribution')
    plt.savefig('./EDA/Guardian_plot.png', bbox_inches='tight')

    #travel time
    # (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
    f, ax = plt.subplots()
    figure = sns.countplot(x='traveltime', data=data, order=[1, 2, 3, 4])
    ax = ax.set(ylabel="Count", xlabel="travel time")
    figure.grid(False)
    plt.title('Travel Time Distribution')
    plt.savefig('./EDA/travel_time_plot.png', bbox_inches='tight')

    #study time plot
    # (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    f, ax = plt.subplots()
    figure = sns.countplot(x='studytime', data=data, order=[1, 2, 3, 4])
    ax = ax.set(ylabel="Count", xlabel="study time")
    figure.grid(False)
    plt.title('Study Time Distribution')
    plt.savefig('./EDA/Study_time_plot.png', bbox_inches='tight')

    #failures plot
    # (numeric: n if 1<=n<3, else 4)
    f, ax = plt.subplots()
    figure = sns.countplot(x='failures', data=data, order=[0, 1, 2, 3])
    ax = ax.set(ylabel="Count", xlabel="failures")
    figure.grid(False)
    plt.title('failures Distribution')
    plt.savefig('./EDA/failures_plot.png', bbox_inches='tight')

    #school support plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='schoolsup', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="School Support")
    figure.grid(False)
    plt.title('School Support Distribution')
    plt.savefig('./EDA/school_support_plot.png', bbox_inches='tight')

    #family support plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='famsup', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="Family Support")
    figure.grid(False)
    plt.title('Family Support Distribution')
    plt.savefig('./EDA/Family_support_plot.png', bbox_inches='tight')

    #paid classes plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='paid', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="extra paid classes")
    figure.grid(False)
    plt.title('Extra paid classes Distribution')
    plt.savefig('./EDA/paid_claases_plot.png', bbox_inches='tight')

    #extracurricular plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='activities', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="extra-curricular activities")
    figure.grid(False)
    plt.title('extra-curricular activities Distribution')
    plt.savefig('./EDA/extracurricular_plot.png', bbox_inches='tight')

    #####CLOSE FIG TO SAVE MEMORY
    plt.close('all')

    #attend nursery plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='nursery', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="attended nursery")
    figure.grid(False)
    plt.title('attended nursery Distribution')
    plt.savefig('./EDA/attend_nursery_plot.png', bbox_inches='tight')

    #higher edu plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='higher', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="wants to take higher education")
    figure.grid(False)
    plt.title('Students who want to take higher education Distribution')
    plt.savefig('./EDA/higher_education_plot.png', bbox_inches='tight')

    #Internet access at home plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='internet', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="Internet access at home")
    figure.grid(False)
    plt.title('Internet access at home Distribution')
    plt.savefig('./EDA/internet_plot.png', bbox_inches='tight')

    #relationship romantic plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='romantic', data=data, order=['yes', 'no'])
    ax = ax.set(ylabel="Count", xlabel="With a romantic relationship")
    figure.grid(False)
    plt.title('Students with a romantic relationship Distribution')
    plt.savefig('./EDA/rom_relationship_plot.png', bbox_inches='tight')


    #family relationship plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='famrel', data=data, order=[1, 2, 3, 4, 5])
    ax = ax.set(ylabel="Count", xlabel="family relationship")
    figure.grid(False)
    plt.title('family relationship Distribution')
    plt.savefig('./EDA/fam_relationship_plot.png', bbox_inches='tight')


    #free time plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='freetime', data=data, order=[1, 2, 3, 4, 5])
    ax = ax.set(ylabel="Count", xlabel="Freetime")
    figure.grid(False)
    plt.title('Free time Distribution')
    plt.savefig('./EDA/free_time_plot.png', bbox_inches='tight')

    #goin out plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='goout', data=data, order=[1, 2, 3, 4, 5])
    ax = ax.set(ylabel="Count", xlabel="Going Out")
    figure.grid(False)
    plt.title('Going Out Distribution')
    plt.savefig('./EDA/Going_out_plot.png', bbox_inches='tight')

    #alcol consumption plot
    f, ax = plt.subplots()
    figure = sns.countplot(x='Dalc', data=data, order=[1, 2, 3, 4, 5])
    ax = ax.set(ylabel="Count", xlabel="Working")
    figure.grid(False)
    plt.title('Working day alcohol consumption Distribution')
    plt.savefig('./EDA/WDAY_alcohol_consumption_plot.png', bbox_inches='tight')
    f, ax = plt.subplots()
    figure = sns.countplot(x='Walc', data=data, order=[1, 2, 3, 4, 5])
    ax = ax.set(ylabel="Count", xlabel="Weekends")
    figure.grid(False)
    plt.title('Weekend alcohol consumption Distribution')
    plt.savefig('./EDA/WEND_alcohol_consumption_plot.png', bbox_inches='tight')

    # convert finalscore to categorical variable
    data = pd.read_csv('./student.csv', sep=';')
    data['FinalGrade'] = 'na'
    data.loc[(data['G3'] >= 18) & (data['G3'] <= 20), 'FinalGrade'] = 'Excellent'
    data.loc[(data['G3'] >= 15) & (data['G3'] <= 17), 'FinalGrade'] = 'Good'
    data.loc[(data['G3'] >= 11) & (data['G3'] <= 14), 'FinalGrade'] = 'Satisfactory'
    data.loc[(data['G3'] >= 6) & (data['G3'] <= 10), 'FinalGrade'] = 'Poor'
    data.loc[(data['G3'] >= 0) & (data['G3'] <= 5), 'FinalGrade'] = 'Failure'

    #####CLOSE FIG TO SAVE MEMORY
    plt.close('all')

    # relationship status
    perc = (lambda col: col / col.sum())
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    relationship_index = pd.crosstab(index=data.FinalGrade, columns=data.romantic)
    romantic_index = relationship_index.apply(perc).reindex(index)
    romantic_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Relationship Status', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_Relationshipstatus.png', bbox_inches='tight')

    # fam sup status
    perc = (lambda col: col / col.sum())
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    relationship_index = pd.crosstab(index=data.FinalGrade, columns=data.famsup)
    romantic_index = relationship_index.apply(perc).reindex(index)
    romantic_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By family support Status', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_family support.png', bbox_inches='tight')

    # grade by age plot
    perc = (lambda col: col / col.sum())
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    relationship_index = pd.crosstab(index=data.FinalGrade, columns=data.age)
    romantic_index = relationship_index.apply(perc).reindex(index)
    romantic_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By age ', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_Age.png', bbox_inches='tight')

    # Dalc - workday alcohol consumption
    alcohol_index = pd.crosstab(index=data.FinalGrade, columns=data.Dalc)
    workday_alcohol_index = alcohol_index.apply(perc).reindex(index)
    workday_alcohol_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By workday alcohol Consumption', fontsize=20)
    plt.ylabel('Percentage of Students ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_workday_alchol.png', bbox_inches='tight')

    # Walc - weekday alcohol consumption
    alcohol_index = pd.crosstab(index=data.FinalGrade, columns=data.Walc)
    weekend_alcohol_index = alcohol_index.apply(perc).reindex(index)
    weekend_alcohol_index.plot.bar(colormap='winter', fontsize=16, figsize=(14, 8))
    plt.title('Grade By weekend alcohol Consumption', fontsize=20)
    plt.ylabel('Percentage of Students ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_weekend_alchol.png', bbox_inches='tight')

    # health - current health status
    health_index = pd.crosstab(index=data.FinalGrade, columns=data.health)
    Overall_health_index = health_index.apply(perc).reindex(index)
    Overall_health_index.plot.bar(colormap='summer', fontsize=16, figsize=(14, 8))
    plt.title('Grade By Overall health', fontsize=20)
    plt.ylabel('Percentage of Students ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_overall_health.png', bbox_inches='tight')

    # goout - going out with friends (numeric: from 1 - very low to 5 - very high)
    goout_index = pd.crosstab(index=data.FinalGrade, columns=data.goout)
    Overall_goout_index = goout_index.apply(perc).reindex(index)
    Overall_goout_index.plot.bar(colormap='jet', fontsize=16, figsize=(14, 8))
    plt.title('Grade By Going Out frequency', fontsize=20)
    plt.ylabel('Percentage of Students ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_going_out.png', bbox_inches='tight')

    #absences
    data['Regularity'] = 'na'
    data.loc[(data.absences >= 0) & (data.absences <= 9), 'Regularity'] = 'Always Regular'
    data.loc[(data.absences >= 10) & (data.absences <= 29), 'Regularity'] = 'Mostly Regular'
    data.loc[(data.absences >= 30) & (data.absences <= 49), 'Regularity'] = 'Regular'
    data.loc[(data.absences >= 50) & (data.absences <= 79), 'Regularity'] = 'Irregular'
    data.loc[(data.absences >= 80) & (data.absences <= 93), 'Regularity'] = 'Highly Irregular'

    #grade by absences
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    absences = pd.crosstab(index=data.FinalGrade, columns=data.Regularity)
    absences = perc(absences)
    absences.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade by students regularity', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_regularity.png', bbox_inches='tight')

    # 31 G1 - first period grade (numeric: from 0 to 20)
    # 31 G2 - second period grade (numeric: from 0 to 20)
    # 32 G3 - final grade (numeric: from 0 to 20, output target)
    data['Grade1'] = 'na'
    data.loc[(data.G1 >= 18) & (data.G1 <= 20), 'Grade1'] = 'Excellent'
    data.loc[(data.G1 >= 15) & (data.G1 <= 17), 'Grade1'] = 'Good'
    data.loc[(data.G1 >= 11) & (data.G1 <= 14), 'Grade1'] = 'Satisfactory'
    data.loc[(data.G1 >= 6) & (data.G1 <= 10), 'Grade1'] = 'Poor'
    data.loc[(data.G1 >= 0) & (data.G1 <= 5), 'Grade1'] = 'Failure'

    data['Grade2'] = 'na'
    data.loc[(data.G2 >= 18) & (data.G2 <= 20), 'Grade2'] = 'Excellent'
    data.loc[(data.G2 >= 15) & (data.G2 <= 17), 'Grade2'] = 'Good'
    data.loc[(data.G2 >= 11) & (data.G2 <= 14), 'Grade2'] = 'Satisfactory'
    data.loc[(data.G2 >= 6) & (data.G2 <= 10), 'Grade2'] = 'Poor'
    data.loc[(data.G2 >= 0) & (data.G2 <= 5), 'Grade2'] = 'Failure'

    #grade by internet
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    internet_index = pd.crosstab(index=data.FinalGrade, columns=data.internet)
    internet_index = internet_index.apply(perc).reindex(index)
    internet_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By internet Status', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_internet_status.png', bbox_inches='tight')

    #####CLOSE FIG TO SAVE MEMORY
    plt.close('all')

    #grade by studytime
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    studytime_index = pd.crosstab(index=data.FinalGrade, columns=data.studytime)
    studytime_index = studytime_index.apply(perc).reindex(index)
    studytime_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Study Time', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_study_time.png', bbox_inches='tight')

    #grade by gender
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    gender_index = pd.crosstab(index=data.FinalGrade, columns=data.sex)
    gender_index = gender_index.apply(perc).reindex(index)
    gender_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By gender', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_gender.png', bbox_inches='tight')

    #grade by location
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    Location_index = pd.crosstab(index=data.FinalGrade, columns=data.address)
    Location_index = Location_index.apply(perc).reindex(index)
    Location_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Location', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_location.png', bbox_inches='tight')

    # grade by parent job
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    Mothers_index = pd.crosstab(index=data.FinalGrade, columns=data.Mjob)
    Mothers_index = Mothers_index.apply(perc).reindex(index)
    Mothers_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Mother Job', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_mothers_job.png', bbox_inches='tight')

    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    Fathers_index = pd.crosstab(index=data.FinalGrade, columns=data.Fjob)
    Fathers_index = Fathers_index.apply(perc).reindex(index)
    Fathers_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Father Job', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_fathers_job.png', bbox_inches='tight')

    #grade by parent edu
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    Mothers_index = pd.crosstab(index=data.FinalGrade, columns=data.Medu)
    Mothers_index = Mothers_index.apply(perc).reindex(index)
    Mothers_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Mother Education', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_mothers_edu.png', bbox_inches='tight')

    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    Fathers_index = pd.crosstab(index=data.FinalGrade, columns=data.Fedu)
    Fathers_index = Fathers_index.apply(perc).reindex(index)
    Fathers_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Father Education', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_fathers_edu.png', bbox_inches='tight')

    #grade by edu status plot
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    higher_index = pd.crosstab(index=data.FinalGrade, columns=data.higher)
    higher_index = higher_index.apply(perc).reindex(index)
    higher_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By higher education Status', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_mhigher_education_status.png', bbox_inches='tight')

    #Grade By Parental Status plot
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    status_index = pd.crosstab(index=data.FinalGrade, columns=data.Pstatus)
    status_index = status_index.apply(perc).reindex(index)
    status_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By Parental Status', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_Parental_status.png', bbox_inches='tight')

    #grade by Failures plot
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    status_index = pd.crosstab(index=data.FinalGrade, columns=data.failures)
    status_index = status_index.apply(perc).reindex(index)
    status_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By failures', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_Failure_status.png', bbox_inches='tight')

    #grade by freetime
    index = ['Failure', 'Poor', 'Satisfactory', 'Good', 'Excellent']
    status_index = pd.crosstab(index=data.FinalGrade, columns=data.freetime)
    status_index = status_index.apply(perc).reindex(index)
    status_index.plot.bar(fontsize=16, figsize=(14, 8))
    plt.title('Grade By freetime', fontsize=20)
    plt.ylabel('Percentage of Students', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.savefig('./EDA/GRADE-BY_plot/Grade_freetime_status.png', bbox_inches='tight')

    data.to_csv('./input/features.csv', index=False, sep=';')

