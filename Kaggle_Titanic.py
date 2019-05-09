# -*- coding: utf-8 -*-
"""
Created on Tue May 02 16:31:39 2019

@author: Asim Islam

 ---------------------------------------------------------------------------
Titanic: Machine Learning from Disaster
https://www.kaggle.com/c/titanic
---------------------------------------------------------------------------
Data Dictionary
VariableDefinitionKey survival 
Survival 0 = No, 1 = Yes 
pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd 
sex Sex 
Age Age in years 
sibsp # of siblings / spouses aboard the Titanic parch # of parents / children aboard the Titanic 
ticket Ticket number fare Passenger fare 
cabin Cabin number embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""

#===============================================
#  Import Libraries
#===============================================
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', 80, 'display.max_rows', 50)
#===============================================


############################################################
#   TITANIC workflow:
#       1)  load dataset, dataset overview, data cleaning
#       2)  data visualization (plots)
#       3)  correlations
#       4)  machine learning
############################################################



################################################
#  1)  Load Dataset, Dataset Overview, Data Cleaning
################################################
#===============================================
#  Load Dataset
#===============================================
#  NOTE: test.csv does not provide 'Survived' information
df_TRN=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/train.csv')
df_TST=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/test.csv')

#  full data
df_ALL = pd.concat([df_TRN,df_TST])
df_ALL['Embarked'].value_counts()
df_ALL['Embarked'].groupby(df_ALL['Pclass']).value_counts()

df_TRN['Embarked'].groupby(df_TRN['Survived']).value_counts()
df_TRN['Pclass'].groupby(df_TRN['Survived']).value_counts()
df_TRN['SibSp'].groupby(df_TRN['Survived']).value_counts()
df_TRN['Parch'].groupby(df_TRN['Survived']).value_counts()
df_TRN['Sex'].groupby(df_TRN['Survived']).value_counts()

df_TRN['Survived'].groupby(df_TRN['Sex']).value_counts()
#===============================================


#===============================================
#  Dataset Overview
#===============================================
#  categorize the dataset
"""
Categorical:
    Nominal -   variables that have two or more categories, but 
    which do not have an intrinsic order
    Dichotomous - Nominal variable with only two categories
    Ordinal -   variables that have two or more categories just 
    like nominal variables. Only the categories can also be 
    ordered or ranked
"""
#  CATEGORICAL:
#  - Nominal:      Cabin, Embarked
#  - Dichotomous:  Sex
#  - Ordinal:      Pclass
#  NUMERIC:
#  - Discrete:     PassengerID, SibSp, Parch, Survived
#  - Continuous:   Age, Fare
#  TEXT VARIABLE:  Ticket, Name
#===============================================
df_TRN.describe(include='O')
df_TST.describe(include='O')
#===============================================


#===============================================
#  Data Cleaning
#   1)  Identify NULLs and Duplicates
#   2)  Fill Nulls for Embarked, Fare and Age
#   3)  Drop columns that are not needed
#===============================================
#-----------------------------------------------
#   Identify NULLs
#-----------------------------------------------
#  heatmap of null values
fig = plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.title('df_TRN - "nulls\"')
sns.heatmap(df_TRN.isnull())
fig.add_subplot(122)
plt.title('df_TST - "nulls\"')
sns.heatmap(df_TST.isnull())
plt.show()


#  3rd method for Percentage of null values - BEST
for i in [df_TRN,df_TST]:
    nulls = i.isnull().sum().sort_values(ascending = False)
    prcet = round(nulls/len(i)*100,2)
    i.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])
print(pd.concat([df_TRN.null,df_TST.null], axis=1,keys=['TRAIN Data', 'TEST Data']))


df_TRN.isnull().sum().sort_values()  #  null values:      Age & Cabin
df_TST.isnull().sum().sort_values()  #  null values:      Age, Cabin & Fare

df_TRN.duplicated().sum()            #  duplicate values: none
df_TST.duplicated().sum()            #  duplicate values: none


#  1st method for Percentage of null values:
total = df_TRN.isnull().sum().sort_values(ascending = False)
percent = round(total/len(df_TRN)*100,2)
print('dataset:  df_TRN\n',pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent']).head(4))

total = df_TST.isnull().sum().sort_values(ascending = False)
percent = round(total/len(df_TST)*100,2)
print('dataset:  df_TST\n',pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent']).head(4))

#  2nd method for Percentage of null values
print('Percentage missing')
for i in sorted(['Age','Cabin','Fare','Embarked']):
    print('{0:10s} {1:7.2%}\ttrain'.format(i,df_TRN[i].isnull().sum()/len(df_TRN)))
    print('{0:10s} {1:7.2%}\ttest'.format(i,df_TST[i].isnull().sum()/len(df_TST)))

df_TRN.isnull().sum().sort_values()  #  null values:      Age & Cabin
df_TST.isnull().sum().sort_values()  #  null values:      Age, Cabin & Fare

#  3rd method for Percentage of null values - BEST
for i in [df_TRN,df_TST]:
    nulls = i.isnull().sum().sort_values(ascending = False)
    prcet = round(nulls/len(i)*100,2)
    i.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])
print(pd.concat([df_TRN.null,df_TST.null], axis=1,keys=['TRAIN Data', 'TEST Data']))
#-----------------------------------------------

"""#======== testing out print formatting ============
for x in range(0,10):
    print("numbers are {0:2d} {1:4d} {2:15d}".format(x,x*x,x*x*x))
"""#==================================================


#-----------------------------------------------
#  Fill Nulls for Embarked, Fare and Age
#       Embarked     - fill with most common location
#       Fare         - fill with mean value of Fare
#       Age          - fill with age per class per sex
#-----------------------------------------------
#  Two Embarked values are missing in train data, fill with 'S'
df_TRN.groupby(df_TRN['Embarked']).sum()
df_TRN['Embarked'].fillna('S', inplace=True)  # fill most mode

#  One Fare is missing in test data, fill it with mean Fare value
df_TST['Fare'].fillna(df_TST['Fare'].mean(), inplace=True)  # fill mean


#  determine mean ages for each class and sex
#  plot of age, sex, pclass
fig = plt.figure(figsize=(8,4))
fig.add_subplot(121)
plt.title('TRAIN - Age/Sex per Passenger Class')
sns.barplot(data=df_TRN, x='Pclass',y='Age',hue='Sex')
fig.add_subplot(122)
plt.title('TEST - Age/Sex per Passenger Class')
sns.barplot(data=df_TST, x='Pclass',y='Age',hue='Sex')
plt.show()


#  calculate age per pclass and sex
#  training - mean Age per Pclass and Sex
meanAgeTrnMale = round(df_TRN[(df_TRN['Sex'] == "male")]['Age'].groupby(df_TRN['Pclass']).mean(),2)
meanAgeTrnFeMale = round(df_TRN[(df_TRN['Sex'] == "female")]['Age'].groupby(df_TRN['Pclass']).mean(),2)
#print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale], axis = 1,keys= ['Male','Female']))

#  test - - mean Age per Pclass and Sex
#  MAY DELETE - may not fill out test data
meanAgeTstMale = round(df_TST[(df_TST['Sex'] == "male")]['Age'].groupby(df_TST['Pclass']).mean(),2)
meanAgeTstFeMale = round(df_TST[(df_TST['Sex'] == "female")]['Age'].groupby(df_TST['Pclass']).mean(),2)
#print(pd.concat([meanAgeTstMale, meanAgeTstFeMale], axis = 1,keys= ['Male','Female']))

print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale,meanAgeTstMale, meanAgeTstFeMale], axis = 1,keys= ['Male-TRN','Female-TRN','Male-TST','Female-TST']))



#  define function APS to fill Age NaN for training data
def age_fillna_TRN(APStrn):
    Age     = APStrn[0]
    Pclass  = APStrn[1]
    Sex     = APStrn[2]
    
    if pd.isnull(Age):
        if Sex == 'male':
            if Pclass == 1:
                return 41.28
            if Pclass == 2:
                return 30.74
            if Pclass == 3:
                return 26.51

        if Sex == 'female':
            if Pclass == 1:
                return 34.61
            if Pclass == 2:
                return 28.72
            if Pclass == 3:
                return 21.75
    else:
        return Age

#  define function APS to fill Age NaN for test data
def age_fillna_TST(APStst):
    Age     = APStst[0]
    Pclass  = APStst[1]
    Sex     = APStst[2]
    
    if pd.isnull(Age):
        if Sex == 'male':
            if Pclass == 1:
                return 40.52
            if Pclass == 2:
                return 30.94
            if Pclass == 3:
                return 24.53

        if Sex == 'female':
            if Pclass == 1:
                return 41.33
            if Pclass == 2:
                return 24.38
            if Pclass == 3:
                return 23.07
    else:
        return Age


df_TRN['Age'] = df_TRN[['Age','Pclass','Sex']].apply(age_fillna_TRN,axis=1)
df_TST['Age'] = df_TST[['Age','Pclass','Sex']].apply(age_fillna_TST,axis=1)
#-----------------------------------------------


#-----------------------------------------------
#  Drop Columns
#  Following columns will be dropped since they add no value to analysis
#  - Cabin:         too many missing values
#  - Name:          text values
#  - Ticket:        text values
#-----------------------------------------------
df_TRN = df_TRN.drop(['Cabin','Name','Ticket'], axis=1)
df_TST = df_TST.drop(['Cabin','Name','Ticket'], axis=1)


# Final check for NULLs
#  3rd method for Percentage of null values - BEST
for i in [df_TRN,df_TST]:
    nulls = i.isnull().sum().sort_values(ascending = False)
    prcet = round(nulls/len(i)*100,2)
    i.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])
print(pd.concat([df_TRN.null,df_TST.null], axis=1,keys=['TRAIN Data', 'TEST Data']))

#df_TRN=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/train.csv')
#df_TST=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/test.csv')
#-----------------------------------------------



################################################
#   2) Data Visualization - PLOTS
#  CATEGORICAL:
#  - Nominal:      Embarked
#  - Dichotomous:  Sex
#  - Ordinal:      Pclass
#  NUMERIC:
#  - Discrete:     SibSp, Parch, Survived
#  - Continuous:   Age, Fare
################################################
#-----------------------------------------------
#  Plot CATEGORICAL and NUMERIC-discrete
#  ['Embarked','Sex','Pclass','SibSp','Parch',]
#-----------------------------------------------
fig = plt.figure(figsize=(10,16))
plotNum  = 1     # initialize plot number
plotList = ['Embarked','Pclass','SibSp','Parch','Sex']

for i in plotList:
    fig.add_subplot(3,2,plotNum)
    plt.title('Survival Rate per %s' %i, fontsize=14)
    plt.xlabel(i, fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.axis('auto')
    plt.grid()
    sns.barplot(df_TRN[i],df_TRN['Survived'],palette='PuBuGn')
    plotNum = plotNum + 1
    #print(df_TRN[[i,'Survived']].groupby(df_TRN[i]).mean())
plt.show()
#-----------------------------------------------


#-----------------------------------------------
#  Plot NUMERIC-continous
#  ['Age','Fare']
#-----------------------------------------------
#  age & fare - facetgrid  - option 1, BEST
fig = plt.figure(figsize=(10,4))
fig.add_subplot(121)
g1 = sns.FacetGrid(df_TRN, col='Survived')
g1.map(plt.hist, 'Age', bins=20)
fig.add_subplot(122)
g2 = sns.FacetGrid(df_TRN, col='Survived')
g2.map(plt.hist, 'Fare', bins=20)


#  age & fare - lineplot  - option 2
fig = plt.figure(figsize=(10,4))
plotNum  = 1     # initialize plot number

for i in df_TRN[['Age','Fare']]:
    fig.add_subplot(1,2,plotNum)
    plt.title('Survival Rate per %s' %i, fontsize=14)
    plt.xlabel(i, fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.axis('auto')
    #plt.xlim(0,100,10)
    plt.grid()
    sns.lineplot(data=df_TRN,x=df_TRN[i],y=df_TRN['Survived'])
    #sns.countplot(data=df_TRN,x=df_TRN[i])
    plotNum = plotNum + 1
plt.show()

#  age & fare - scatterplot  - option 3
#  Survival - Age and Fare
Surv0 = df_TRN[(df_TRN['Survived'] == 0)]
Surv1 = df_TRN[(df_TRN['Survived'] == 1)]

fig = plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.title('Age vs Fare - Survived=0', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.axis('auto')
plt.grid()
sns.scatterplot(data=Surv0,x=Surv0.Age,y=Surv0.Fare,palette='seismic')#,hue=Surv0.Survived)
fig.add_subplot(122)
plt.title('Age vs Fare - Survived=1', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.axis('auto')
plt.grid()
sns.scatterplot(data=Surv1,x=Surv1.Age,y=Surv1.Fare)#,hue=Surv0.Survived)
plt.show()
#-----------------------------------------------


#?????????????????
"""
#  plot and table together
fig = plt.figure(figsize=(10,4))
fig.add_subplot(121)
plt.title('df_TRN - "nulls\"')
sns.heatmap(df_TRN.isnull())
fig.add_subplot(122)
plt.title('df_TST - "nulls\"')
#fig, ax = plt.subplots()
plt.axis('off')
plt.table(cellText=aaa.values,colLabels=aaa.columns,loc='center')
plt.show()
"""
#?????????????????


################################################
#  CORRELATION
#    - map Sex and Embared as numeric
#    - overall
#    - male and female
################################################
# Print all survival rates  <--  CORRELATION
for i in df_TRN[['Pclass','Sex','SibSp','Parch','Embarked']]:
    print(df_TRN[[i, 'Survived']].groupby(i, as_index=False).mean().sort_values(by='Survived', ascending=False),"\n")


#-----------------------------------------------
#  MAPPING - Sex, Embarked as NUMERIC
#-----------------------------------------------
df_TRN['Sex'] = df_TRN['Sex'].map({'female':0,'male':1})
df_TRN['Embarked'] = df_TRN['Embarked'].map({'S':0,'C':1,'Q':2})
df_TST['Sex'] = df_TST['Sex'].map({'female':0,'male':1})          # for modeling
df_TST['Embarked'] = df_TST['Embarked'].map({'S':0,'C':1,'Q':2})  # for modeling
#-----------------------------------------------


#-----------------------------------------------
#  Correlations - OVERALL
#-----------------------------------------------
plt.figure(figsize=(8,8))
plt.title('Survived Correlation - OVERALL', fontsize=14)
sns.heatmap(df_TRN.corr(), annot=True, fmt='.2f', square=True, cmap = 'Greens_r')
df_TRN.corr()['Survived'].sort_values(ascending=False)
#-----------------------------------------------


#-----------------------------------------------
#  Correlation FEMALE - filter dataframe for male/female
dataFemale = df_TRN[(df_TRN['Sex'] == 0)]                       # female
dataFemaleCorr = dataFemale.drop(["Sex"], axis=1).corr()    # female corr
corrF = dataFemaleCorr['Survived'].sort_values(ascending=False)

#  Correlation MALE - filter dataframe for male/female
dataMale   = df_TRN[(df_TRN['Sex'] == 1)]                       # male
dataMaleCorr = dataMale.drop(["Sex"], axis=1).corr()        # male corr
corrM = dataMaleCorr['Survived'].sort_values(ascending=False)


#  Correlation heatmaps for FEMALE/MALE
fig = plt.figure(figsize=(12,6))
fig.add_subplot(121)
plt.title('Survived Correlation - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
fig.add_subplot(122)
plt.title('Survived Correlation - FEMALE ', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
plt.show()

#  Correlation - sorted for both male/female
corrALL = pd.DataFrame(columns = ['MALE','correlation-m','FEMALE','correlation-f'])
corrALL['MALE']   = corrM.index
corrALL['correlation-m'] = corrM.values
corrALL['FEMALE'] = corrF.index
corrALL['correlation-f'] = corrF.values
print(corrALL)
#-----------------------------------------------
############################################################




################################################
#  MACHINE LEARNING
#   - Split Dataset Into Train and Test
#   - Define the Models
#   - Evaluate the Models
################################################
#-----------------------------------------------
#  Split Dataset Into Train and Test
#-----------------------------------------------
validation_size = 0.20
seed = 7
scoring='accuracy'

X = df_TRN.drop(['Survived'], axis = 1)
y = df_TRN['Survived']
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, 
        test_size = validation_size, 
        random_state = seed)
#-----------------------------------------------


#-----------------------------------------------
#  Define the Models
"""
Evaluate different algorithms:
------------------------------
Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN)
Classification and Regression Trees (CART)
Gaussian Naive Bayes (NB)
Support Vector Machines (SVM)
Random Forest (RF)
"""
#-----------------------------------------------
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('DT  ', DecisionTreeClassifier()))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto')))      # supportVector
models.append(('RF  ', RandomForestClassifier()))

models.sort()
#-----------------------------------------------


#-----------------------------------------------
#  Evaluate the Models
#   - cross-validation
#   - accuracy score
#   - confusion matrix
#   - classification report
#-----------------------------------------------
#  Evaluate the Models:
#   - cross-validation
results = []
names = []
modelDF = pd.DataFrame(columns=['model','CV-mean','CV-std'])
countDF = 0

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    #msg = "%s:  %f  (%f)" % (name, cv_results.mean(), cv_results.std())
    msg = ("{0:s}:  {1:3.5f}  ({2:3.5f})".format(name, cv_results.mean(), cv_results.std()))
    print(msg)
    modelDF.loc[countDF]=[name, cv_results.mean(), cv_results.std()]
    countDF = countDF + 1

#  Cross-Validation boxplot
fig = plt.figure(figsize=(6,4))
#fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.title('Algorithm Comparison', fontsize=14)
plt.xlabel('algorithm', fontsize=12)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# sort on Cross-Validation
print("\nCROSS-VALIDATION:\n{}".format(modelDF.sort_values(['CV-mean'], ascending=False)))


#  Evaluate the Models:
#   - Accuracy Score
#   - Confusion Matrix
#   - Classification Report
results2 = []
names2 = []

for name, model in models:
    modelML = model
    modelML.fit(X_train,y_train)
    modelPredict = modelML.predict(X_test)
    results2.append(round(accuracy_score(y_test,modelPredict)*100,2))
    names2.append(name)
    print("\n=====  {0:3s} =====\nAccuracy Score: \t{1:3.2%}".format(name,accuracy_score(y_test,modelPredict)))
    print("Confusion Matrix:\n{0:}".format(confusion_matrix(y_test,modelPredict)))
    print("Classification Report: \n{0:}".format(classification_report(y_test,modelPredict)))

#  accuracy score barplot
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
plt.title('Algorithm Accuracy Score', fontsize=14)
plt.xlabel('Algorithm', fontsize=12)
plt.ylabel('Accuracy Score (%)', fontsize=12)
plt.bar(names2,results2, color='rgbkymc')
ax.set_xticklabels(names2)
plt.show()

# sort on Accuracy Score
dfAS = pd.DataFrame(list(zip(names2,results2)), columns=['Algorithm','Score (%)'])
print("\nAccuracy Scores:\n{}".format(dfAS.sort_values(['Score (%)'], ascending=False)))
#####################################################


#####################################################
#  Submission file
#####################################################
RF = RandomForestClassifier()       #  best model
RF.fit(X_test,y_test)               #  fit
SF = RF.predict(df_TST)             #  predictions
SF = pd.DataFrame(SF, columns=['Survived'])
SF_TST = pd.concat([df_TST, SF], axis=1, join='inner')
SF_final = SF_TST[['PassengerId','Survived']]
SF_final.info()
SF_final.to_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/predictions.csv', index=False)
#####################################################