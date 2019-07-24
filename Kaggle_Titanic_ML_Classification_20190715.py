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

# Machine Learning
#  Normalize data
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn import model_selection

# Machine Learning
from sklearn.linear_model import LogisticRegression
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Machine Learning Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
#%matplotlib inline
#import os
#print(os.listdir("../input"))
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# color palletes
sns.palplot(sns.color_palette())
sns.palplot(sns.color_palette("Greens",5))
sns.palplot(sns.color_palette("BrBG", 4))
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
df_ALL['Survived'].groupby(df_ALL['Sex']).value_counts()

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
#  - Discrete:     PassengerId, SibSp, Parch, Survived
#  - Continuous:   Age, Fare
#  TEXT VARIABLE:  Ticket, Name
#===============================================
#  Change type to 'object' for categorical data
df_TRN.info()
df_TRN.PassengerId = df_TRN.PassengerId.astype(object)
df_TRN.Survived = df_TRN.Survived.astype(object)
df_TRN.Pclass = df_TRN.Pclass.astype(object)
df_TRN.SibSp = df_TRN.SibSp.astype(object)
df_TRN.Parch = df_TRN.Parch.astype(object)

df_TRN.Name = df_TRN.Name.astype(str)
df_TRN.Ticket = df_TRN.Ticket.astype(str)

df_TRN.info()

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



df_TRN.isnull().sum().sort_values()  #  null values:      Age & Cabin
df_TST.isnull().sum().sort_values()  #  null values:      Age, Cabin & Fare

df_TRN.duplicated().sum()            #  duplicate values: none
df_TST.duplicated().sum()            #  duplicate values: none

'''
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
'''
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
plt.title('TRAIN - Age/Sex per Class')
ax1 = sns.barplot(data=df_TRN, x='Pclass',y='Age',hue='Sex',ci=None)
for p in ax1.patches:
    ax1.annotate("%.2f" % p.get_height(), xy=(p.get_x(), p.get_height()))
fig.add_subplot(122)
plt.title('TEST - Age/Sex per Class')
ax2 = sns.barplot(data=df_TST, x='Pclass',y='Age',hue='Sex',ci=None)
for p in ax2.patches:
    ax2.annotate("%.2f" % p.get_height(), xy=(p.get_x(), p.get_height()))
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


#  Run function
df_TRN['Age'] = df_TRN[['Age','Pclass','Sex']].apply(age_fillna_TRN,axis=1)
df_TST['Age'] = df_TST[['Age','Pclass','Sex']].apply(age_fillna_TST,axis=1)
#-----------------------------------------------


#-----------------------------------------------
### 3.1.2  CABIN - drop NULLs
#**Cabin** has more then 70% data missing and will be dropped, along with **PassengerId**, **Name** and #**Ticket**, since they add limited value to this analysis:
#* Cabin:         too many missing values
#* PassengerId:   unique integer value
#* Name:          unique text values
#* Ticket:        unique text values
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
#  Plot CATEGORICAL
#  ['Embarked','Sex','Pclass','SibSp','Parch',]
#-----------------------------------------------
fig = plt.figure(figsize=(10,16))
plotNum  = 1     # initialize plot number
plotList = ['Sex','Pclass','Embarked','SibSp','Parch']

for i in plotList:
    fig.add_subplot(3,2,plotNum)
    plt.title('Survival Rate per %s' %i, fontsize=14)
    plt.xlabel(i, fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.axis('auto')
    plt.grid()
    #sns.barplot(df_TRN[i],df_TRN['Survived'])#, ci=None)
    sns.swarmplot(data=df_TRN, x=df_TRN[i],y=df_TRN.Age,hue=df_TRN.Survived)
    plotNum = plotNum + 1
    #print(df_TRN[[i,'Survived']].groupby(df_TRN[i]).mean())
plt.show()
#-----------------------------------------------


sns.swarmplot(data=df_TRN, x=df_TRN.Sex,y=df_TRN.Age,hue=df_TRN.Survived)
sns.swarmplot(data=df_TRN, x=df_TRN.Pclass,y=df_TRN.Age,hue=df_TRN.Survived)



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
'''
df_TRN.Survived = df_TRN.Survived.astype('int64')

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
'''

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
#  Plot Age and Fare per Pclass
#-----------------------------------------------
fig = plt.figure(figsize=(12,6))
fig.add_subplot(121)
df_TRN.Age[df_TRN.Pclass == 1].plot(kind='kde')
df_TRN.Age[df_TRN.Pclass == 2].plot(kind='kde')
df_TRN.Age[df_TRN.Pclass == 3].plot(kind='kde')
plt.title('Age Distribution per Class')
plt.xlabel('Age')
plt.legend(('1st Class','2nd Class','3rd Class'), loc='best')
plt.xlim(0,100)

fig.add_subplot(122)
df_TRN.Fare[df_TRN.Pclass == 1].plot(kind='kde')
df_TRN.Fare[df_TRN.Pclass == 2].plot(kind='kde')
df_TRN.Fare[df_TRN.Pclass == 3].plot(kind='kde')
plt.title('Fare Distribution per Class')
plt.xlabel('Fare')
plt.legend(('1st Class','2nd Class','3rd Class'), loc='best')
plt.xlim(0,120)
plt.show()


df_class = pd.DataFrame(columns = {'1st-Age','1st-Fare','2nd-Age','2nd-Fare','3rd-Age','3rd-Fare'})
df_class[['1st-Age','1st-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 1].describe()
df_class[['2nd-Age','2nd-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 2].describe()
df_class[['3rd-Age','3rd-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 3].describe()
df_class = df_class[['1st-Age','2nd-Age','3rd-Age','1st-Fare','2nd-Fare','3rd-Fare']]
df_class
#-----------------------------------------------

#  Percentage Children per Class
df_TRN.Age[(df_TRN.Pclass == 1) & (df_TRN.Age < 18)].count()/len(df_TRN.Age[df_TRN.Pclass == 1])
df_TRN.Age[(df_TRN.Pclass == 2) & (df_TRN.Age < 18)].count()/len(df_TRN.Age[df_TRN.Pclass == 2])
df_TRN.Age[(df_TRN.Pclass == 3) & (df_TRN.Age < 18)].count()/len(df_TRN.Age[df_TRN.Pclass == 3])

#  Survival Rate per Class
df_TRN.Age[(df_TRN.Pclass == 1) & (df_TRN.Survived == 1)].count()/len(df_TRN.Age[df_TRN.Pclass == 1])
df_TRN.Age[(df_TRN.Pclass == 2) & (df_TRN.Survived == 1)].count()/len(df_TRN.Age[df_TRN.Pclass == 2])
df_TRN.Age[(df_TRN.Pclass == 3) & (df_TRN.Survived == 1)].count()/len(df_TRN.Age[df_TRN.Pclass == 3])



#####################################################
#  MODELING
#####################################################
df_origTRN = df_TRN.copy(deep=True)
#df_TRN = df_origTRN.copy(deep=True)

#=================================================
#  One Hot Encoding
#=================================================
#  Set categorical attributes as type 'object'
for i in ['Pclass','SibSp','Parch','Sex','Embarked']:
    df_TRN[i] = df_TRN[i].astype(object)   #  training data
    df_TST[i] = df_TST[i].astype(object)   #  test data

#  one-hot encoding:  training data
#df_OH_TRN = df_TRN[['Pclass','SibSp','Parch','Sex','Embarked','Survived']]
df_OH_TRN = df_TRN[['Pclass','SibSp','Parch','Sex','Embarked']]
df_OH_TRN = pd.get_dummies(df_OH_TRN)       #  one-hot encoding
df_OH_TRN = df_OH_TRN.join(df_TRN[['Age','Fare']])
df_OH_TRN.info()

#  one-hot encoding:  test data
df_OH_TST = df_TST[['Pclass','SibSp','Parch','Sex','Embarked']]
df_OH_TST = pd.get_dummies(df_OH_TST)       #  one-hot encoding
df_OH_TST = df_OH_TST.join(df_TST[['Age','Fare']])
df_OH_TST.info()
#=================================================


#=================================================
#  Normalize the data
#=================================================
normTRN = MinMaxScaler().fit_transform(df_OH_TRN)
normTRN[0:2]
normTST = MinMaxScaler().fit_transform(df_OH_TST)
normTST[0:2]

#  create dataframe with one hot encoding and normalized data
df_finalTRN = pd.DataFrame(normTRN, index=df_OH_TRN.index, columns=df_OH_TRN.columns)
df_finalTRN['Survived'] = df_TRN['Survived']
df_finalTRN['PassengerId'] = df_TRN['PassengerId']
df_finalTRN.info()
df_finalTST = pd.DataFrame(normTST, index=df_OH_TST.index, columns=df_OH_TST.columns)
df_finalTST['PassengerId'] = df_TST['PassengerId']
df_finalTST.info()
#=================================================

#  Survived should be int64 for correlation
#  Corr for Survived is not impacted by one-hot encoding/normalization
df_finalTRN.Survived = df_finalTRN.Survived.astype('int64')

#=================================================
#  Correlation
#=================================================
#df_finalTRN.corr()[['Survived_0','Survived_1']].sort_values(by='Survived_1',ascending=False)
df_finalTRN.corr()['Survived'].sort_values(ascending=False)


#  Correlation FEMALE - filter dataframe for male/female (>0 means female)
dataFemale = df_finalTRN[(df_finalTRN['Sex_female'] > 0)]  #  [-0.73,  1.35]   TODO-check value
dataFemaleCorr = dataFemale.drop(["Sex_female","Sex_male"], axis=1).corr()
corrF = dataFemaleCorr['Survived'].sort_values(ascending=False)

#  Correlation MALE - filter dataframe for male/female (>0 means male)
dataMale   = df_finalTRN[(df_finalTRN['Sex_male'] > 0)]    #  [0.73, -1.35]   TODO check value
dataMaleCorr = dataMale.drop(["Sex_female","Sex_male"], axis=1).corr()
corrM = dataMaleCorr['Survived'].sort_values(ascending=False)
corrM['Parch_6'] = 0  # 3.38e-15.  all numbers will be exp if not set to 0

#  Correlation - sorted for both male/female
corrALL = pd.DataFrame(columns = ['MALE','correlation-m','FEMALE','correlation-f'])
corrALL['MALE']   = corrM.index
corrALL['correlation-m'] = corrM.values
corrALL['FEMALE'] = corrF.index
corrALL['correlation-f'] = corrF.values
print(corrALL)
#=================================================




################################################
#  MACHINE LEARNING
#   - Split Dataset Into Train and Test
#   - Define the Models
#   - Evaluate the Models
################################################
#-----------------------------------------------
#  Split Dataset Into Train and Test
#-----------------------------------------------
seed = 7
X = df_finalTRN.drop(['Survived'], axis = 1)
y = df_finalTRN['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=seed)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
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
modelCV = pd.DataFrame(columns=['model','CV-mean','CV-std','AccuracyScore','F1Score'])
countCV = 0

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    model.fit(X_train,y_train)
    modelPredict = model.predict(X_test)
    accu = accuracy_score(y_test,modelPredict)
    f1Score = f1_score(y_test,modelPredict)
    print("{0:s}:  {1:3.5f}  ({2:3.5f})  {3:.2%}  {3:.2%}  {3:.2%}".format(name,cv_results.mean(),cv_results.std(),accu,f1Score))
    modelCV.loc[countCV]=[name,cv_results.mean(),cv_results.std(),accu,f1Score]
    countCV = countCV + 1

# sort on Cross-Validation
print("\nCROSS-VALIDATION:\n{}".format(modelCV.sort_values(['CV-mean'], ascending=False)))
print("\nACCURACY SCORE:\n{}".format(modelCV.sort_values(['AccuracyScore'], ascending=False)))
print("\nF1 SCORE:\n{}".format(modelCV.sort_values(['F1Score'], ascending=False)))
#=================================================


#=================================================
#  5.5 Classification Model - Random Forest
#=================================================
RF = RandomForestClassifier().fit(X_train,y_train)
RF

#  predict
y_predict = RF.predict(X_test)
y_predict[0:10]
#=================================================


#=================================================
##  6.1 Cross Validation Score<a id="eval_cv"></a>  
#=================================================
#from sklearn.model_selection import cross_val_score

print(cross_val_score(RF, X_train, y_train, cv=5, scoring='accuracy'))
print('Cross Validation Score (mean):  {:3.4%}'.format(cross_val_score(RF, X_train, y_train, cv=5, scoring='accuracy').mean()))
#=================================================


#=================================================
##  6.2 Accuracy Score<a id="eval_acc"></a>   
#=================================================
#from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)
print('Accuracy Score:  {:3.4%}'.format(accuracy_score(y_test,y_predict)))
#=================================================


#=================================================
##  6.3 F1 Score<a id="eval_f1"></a>   
#=================================================
#from sklearn.metrics import f1_score

f1score = f1_score(y_test, y_predict)
print('F1 Score:  {:3.4%}'.format(f1score))
#=================================================


#=================================================
##  6.4 Confusion Matrix<a id="eval_conf"></a>   
#=================================================
#from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_predict)

sns.heatmap(conf_matrix, annot=True,cmap='Blues',annot_kws={"size": 30})
plt.title("Confusion Matrix, F1 Score: {:3.4%}".format(f1score))
plt.show()

print('True Positive:\t{}'.format(conf_matrix[0,0]))
print('True Negative:\t{}'.format(conf_matrix[0,1]))
print('False Positive:\t{}'.format(conf_matrix[1,0]))
print('False Negative:\t{}'.format(conf_matrix[1,1]))
#=================================================


#=================================================
##  6.5 Receiver Operating Characteristics (ROC) Curve<a id="eval_roc"></a>   
#=================================================
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score

RF.probability = True   # need for predict_proba to work
RF.fit(X_train,y_train)
y_predita = RF.predict_proba(X_test)
y_predita = y_predita[:,1]   # positive values only

ROC_AUC = roc_auc_score(y_test, y_predita)
fpr, tpr, thresholds = roc_curve(y_test, y_predita)

plt.plot([0,1],[0,1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title("ROC Curve, ROC_AUC Score: {:3.4%}".format(ROC_AUC))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#=================================================


#=================================================
##  6.6 Classification Report<a id="eval_class"></a>   
#=================================================
#from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
#=================================================


#=================================================
##  6.7 Log Loss<a id="eval_log"></a>   
#=================================================
#from sklearn.metrics import log_loss

#  predict_proba returns estimates for all classes
y_predict_prob = RF.predict_proba(X_test)
print(y_predict_prob[0:5])

print("\nLog Loss:  {:3.4}".format(log_loss(y_test, y_predict_prob)))
#=================================================




#####################################################
#  Submission file
#####################################################
# Check shape of TEST data
print('Test data shapes must matchin order to \"fit\":')
print('shape X_test:\t\t{}'.format(X_test.shape))
print('shape y_test:\t\t{}'.format(y_test.shape))
print('shape df_finalTST:\t{}'.format(df_finalTST.shape))

#  drop column in test in order to fit the data
df_finalTST = df_finalTST.drop(['Parch_9'], axis=1)


RF = RandomForestClassifier()       #  best model
RF.fit(X_test,y_test)               #  fit
SF = RF.predict(df_finalTST)             #  predictions
SF = pd.DataFrame(SF, columns=['Survived'])
SF_TST = pd.concat([df_finalTST, SF], axis=1, join='inner')
SF_final = SF_TST[['PassengerId','Survived']]
SF_final.info()
SF_final.to_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/titanic/predictions.csv', index=False)
#####################################################