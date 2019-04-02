# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:30:01 2019

@author: Asim Islam

ML Project Heart Disease
---------------------------------------------------------------------------
data SET
---------------------------------------------------------------------------
age         age in years
sex         1 = male; 0 = female
cp          chest pain type
trestbps    resting blood pressure (in mm Hg on admission to the hospital)
chol        serum cholestoral in mg/dl
fbs         (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg     resting electrocardiographic results
thalach     maximum heart rate achieved
exang       exercise induced angina (1 = yes; 0 = no)
oldpeak     ST depression induced by exercise relative to rest
slope       slope of the peak exercise ST segment
ca          number of major vessels (0-3) colored by flourosopy
thal        3 = normal; 6 = fixed defect; 7 = reversable defect
target      0 = no disease, 1 = disease 

source:     https://www.kaggle.com/ronitf/heart-disease-uci
---------------------------------------------------------------------------
    STEPS
  - Import Libraries
  - Load the Dataset
  - Review the Dataset
  - Clean the Dataset
  - Plot the Dataset
  - Calculate and Plot Correlation
  - Split Dataset Into Train and Test
  - Define the Models
  - Evaluate the Models
  - Evaluate the Accuracy of the Models
"""

#####################################################
#  STEP 1:  Import Libraries
#####################################################
# Basic
import pandas as pd
import matplotlib.pyplot as plt

# Other libraries
from sklearn.model_selection import train_test_split
import seaborn as sns

# Machine Learning
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#####################################################


#####################################################
#  STEP 2:  Load the Dataset
#####################################################
data=pd.read_csv('C:/Users/ACER/Desktop/JAVA/MachineLearning/Heart-Disease-Prediction-master/dataset.csv')
#####################################################


#####################################################
#  STEP 3:  Review the Dataset
#####################################################
pd.set_option('display.max_columns', 30, 'display.max_rows', 20)

#  rename columns for better readability
data.rename(columns={
        'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure',
        'chol':'cholestoral','fbs':'fasting_blood_sugar',
        'restecg':'resting_electrocardiographic','thalach':'maximum_heart_rate',
        'exang':'exercise_induced_angina','oldpeak':'ST_depression',
        'slope':'slope_peak_exercise_ST','ca':'number_of_major_vessels'},
    inplace=True)

data.info()         #  dataset size and types
data.describe()     #  statistical summary

data.shape          #  dimensions
data.columns        #  column names
data.head(10)       #  top  10 lines of dataset
data.tail(10)       #  last 10 lines of dataset
data.columns.tolist()
data.age.dtype
#####################################################


#####################################################
#  STEP 4:  Clean the Dataset
#####################################################
#  check for NULL values
data.isnull().sum()                   #  count of null values
data.dropna(inplace=True)             #  drop rows with null values
#  check for DUPLICATES
data.duplicated().sum()               #  count of duplicate values
data.drop_duplicates(inplace = True)  #  drop rows with null values


#  RENAME AND UPDATE VALUES FOR BETTER READABILITY/PLOTTING
#  Modify nominal columns values for better interpretation
#  (refer to dataset for values)
#  dataPlot     dataframe, for plotting
#  data         dataframe, for calculations

# make copy of dataframe for plotting
dataPlot = data.copy(deep=True)   # errors on Kaggle


# map values
dataPlot['sex'] = dataPlot['sex'].map({0:'female', 1:'male'})
dataPlot['chest_pain_type'] = dataPlot['chest_pain_type'].map({
        0:'typical angina', 1:'atypical angina',
        2:'non-anginal',    3:'asymptomatic'})
dataPlot['fasting_blood_sugar'] = dataPlot['fasting_blood_sugar'].map({
        0:'> 120 mg/dl', 1:'< 120 mg/dl'})
dataPlot['resting_electrocardiographic'] = dataPlot['resting_electrocardiographic'].map({
        0:'normal', 1:'ST-T wave abnormality', 2:'ventricular hypertrophy'})
dataPlot['exercise_induced_angina'] = dataPlot['exercise_induced_angina'].map({
        0:'no', 1:'yes'})
dataPlot['slope_peak_exercise_ST'] = dataPlot['slope_peak_exercise_ST'].map({
        0:'upsloping', 1:'flat', 2:'downsloping'})
dataPlot['thal'] = dataPlot['thal'].map({
        0:'normal 0',     1:'normal 1',
        2:'fixed defect', 3:'reversable defect'})
dataPlot['target'] = dataPlot['target'].map({0:'no disease', 1:'disease'})


#  Separate out Ordinal and Nominal data
colnO = []
colnN = []
for i in dataPlot.columns:
    if (len(dataPlot[i].unique())) > 5:
        colnO.append(i)
    else:
        colnN.append(i)
    print(len(dataPlot[i].unique()),"\t",i)

dataNom = dataPlot[colnN];dataNom.info()   #  Nominal data, use histogram plot
colnO.append('target')                     #  add target column to Ordinal
dataOrd = dataPlot[colnO];dataOrd.info()   #  Ordinal data, use scatter plot
#####################################################


#####################################################
#  STEP 5:  Plot the Dataset
#####################################################
#  Plots: Overall, no disease and disease (side by side)
#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
#  assign NOM dataframe for "no disease" and "disease"
diseaseN    = dataNom[(dataPlot['target'] == 'disease')]
no_diseaseN = dataNom[(dataPlot['target'] == 'no disease')]

#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfRow = len(dataNom.columns)
subNumOfCol = 3     # three columns: overall, no disease, disease
subPlotNum  = 1     # initialize plot number

fig = plt.figure(figsize=(16,60))

for i in dataNom.columns:
    # dataPlot
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('Overall', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(dataPlot[i], hue=dataPlot.sex)
    subPlotNum = subPlotNum + 1
    # no_diseaseN
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('No Disease', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(no_diseaseN[i], hue=dataPlot.sex)
    subPlotNum = subPlotNum + 1
    # diseaseN
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('Disease', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(diseaseN[i], hue=dataPlot.sex)
    subPlotNum = subPlotNum + 1

plt.show()


#  Plots: Overall, no disease and disease (side by side)
#  assign ORD dataframe for "no disease" and "disease"
no_diseaseO = dataOrd[(dataPlot['target'] == 'no disease')]
diseaseO    = dataOrd[(dataPlot['target'] == 'disease')]

#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfRow = len(dataNom.columns)-1   #  x='age' in plots, drop column
subNumOfCol = 3     # three columns: overall, no disease, disease
subPlotNum  = 1     # initialize plot number

fig = plt.figure(figsize=(16,50))

for i in dataOrd.columns.drop(["age","target"]):
    # dataPlot
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('Overall', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.scatterplot(data=dataPlot,x='age',y=dataPlot[i],hue='sex')
    subPlotNum = subPlotNum + 1
    # no_diseaseO
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('No Disease', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.scatterplot(data=dataPlot,x='age',y=no_diseaseO[i],hue='sex')
    subPlotNum = subPlotNum + 1
    # diseaseO
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('Disease', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.scatterplot(data=dataPlot,x='age',y=diseaseO[i],hue='sex')
    subPlotNum = subPlotNum + 1

plt.show()
#####################################################


#####################################################
#  STEP 6:  Calculate and Plot Correlation - HEATMAP
#####################################################
#  Correlation OVERALL
dataCorr = data.corr()
plt.figure(figsize=(10,10))
plt.title('correlation Heart Disease - MALE and FEMALE', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Greys')

dataCorr['target'].sort_values(ascending=False)


#  Correlation FEMALE - filter dataframe for male/female
dataFemale = data[(data['sex'] == 1)]                       # female
dataFemaleCorr = dataFemale.drop(["sex"], axis=1).corr()    # female corr
plt.figure(figsize=(10,10))
plt.title('correlation Heart Disease - FEMALE', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')

dataFemaleCorr['target'].sort_values(ascending=False)


#  Correlation MALE - filter dataframe for male/female
dataMale   = data[(data['sex'] == 0)]                       # male
dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr
plt.figure(figsize=(10,10))
plt.title('correlation Heart Disease - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')

dataMaleCorr['target'].sort_values(ascending=False)
#####################################################


#####################################################
#  STEP 7:  Split Dataset Into Train and Test
#####################################################
validation_size = 0.20
seed = 7
scoring='accuracy'

y = data['target']
X = data.drop(['target'], axis = 1)
X_train, X_test, Y_train, Y_test = \
        train_test_split(X, y, 
        test_size = validation_size, 
        random_state = seed)
#####################################################


#####################################################
#  STEP 8:  Define the Models
"""
Evaluate different algorithms:
------------------------------
Logistic Regression (LR)
Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN).
Classification and Regression Trees (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).
Random Forest (RF).
"""
#####################################################
models = []
models.append(('LR  ', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA ', LinearDiscriminantAnalysis()))
models.append(('KNN ', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB  ', GaussianNB()))
models.append(('SVM ', SVC(gamma='auto')))      # supportVector
models.append(('RF  ', RandomForestClassifier()))

models.sort()
#####################################################


#####################################################
#  STEP 9: Evaluate the Models
#    - best: GaussianNB (NB) = 0.829833   (0.070901)
#    - kaggle picks LR LogisticRegression as best
#####################################################
results = []
names = []
modelDF = pd.DataFrame(columns=['model','CV-mean','CV-std'])
countDF = 0

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s:  %f  (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	modelDF.loc[countDF]=[name, cv_results.mean(), cv_results.std()]
	countDF = countDF + 1

# sort on mean Cross-Validation results
modelDF.sort_values(['CV-mean'], ascending=False)


#  Cross-Validation boxplot
fig = plt.figure(figsize=(8,6))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#####################################################


#####################################################
#  STEP 10: Evaluate the Accuracy of the Models
#  kaggle picks LR LogisticRegression(solver='liblinear', multi_class='ovr')))
#####################################################
#  FIT MODEL
#accurModel = GaussianNB()
accurModel = LogisticRegression(solver='liblinear', multi_class='ovr')
accurModel.fit(X_train, Y_train)

#  PREDICT ACCURACY
predictions = accurModel.predict(X_test)
print(accuracy_score(Y_test, predictions)*100,"%")

#  CONFUSION MATRIX
#  [[true positive,  false negative]
#   [false positive, true negative]]
print(confusion_matrix(Y_test, predictions))

#  CLASSIFICATION REPORT
print(classification_report(Y_test, predictions))
#####################################################

