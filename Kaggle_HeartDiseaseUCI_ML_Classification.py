# -*- coding: utf-8 -*-
"""
Created on SAT Jun 29:30:01 2019
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
  
  DATA TYPES:
      Categorical:
          Dichotomous:  sex
          Ordinal:      chest_pain_type, fasting_blood_sugar, resting_electrocardiographic, exercise_induced_angina, slope_peak_exercise_ST, number_of_major_vessels, thal
      Numerical:
          Continous:    age, resting_blood_pressure, cholestoral, maximum_heart_rate, ST_depression
          
        'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure',
        'chol':'cholestoral','fbs':'fasting_blood_sugar',
        'restecg':'resting_electrocardiographic','thalach':'maximum_heart_rate',
        'exang':'exercise_induced_angina','oldpeak':'ST_depression',
        'slope':'slope_peak_exercise_ST','ca':'number_of_major_vessels'},
"""

#####################################################
#  :  Import Libraries
#####################################################
# Basic
import pandas as pd
import matplotlib.pyplot as plt

# Other libraries
from sklearn.model_selection import train_test_split
import seaborn as sns

#  Normalize data
from sklearn.preprocessing import MinMaxScaler

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Machine Learning Evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50, 'display.max_rows', 100)
#%matplotlib inline
#import os
#print(os.listdir("../input"))
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# color palletes
sns.palplot(sns.color_palette())
sns.palplot(sns.color_palette("Greens",5))
sns.palplot(sns.color_palette("BrBG", 4))
#####################################################


#####################################################
#  1:  DATA
#####################################################
#  Read data
df=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/heart.csv')
#=================================================


#  Review data
#=================================================
df.info()         #  dataset size and types
df.describe()     #  statistical summary
df.describe(include='all')     #  statistical summary

df.shape          #  dimensions
df.columns        #  column names
df.head(10)       #  top  10 lines of dataset
df.tail(10)       #  last 10 lines of dataset
df.columns.tolist()
df.age.dtype



#  Clean the Dataset
#=================================================
#  check for NULL values
df.isnull().sum()                   #  count of null values
df.dropna(inplace=True)             #  drop rows with null values
#  check for DUPLICATES
df.duplicated().sum()               #  count of duplicate values
df.drop_duplicates(inplace = True)  #  drop rows with null values



#  Rename columns (attributes) for better readability
#=================================================
df.rename(columns={
        'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure',
        'chol':'cholestoral','fbs':'fasting_blood_sugar',
        'restecg':'resting_electrocardiographic','thalach':'maximum_heart_rate',
        'exang':'exercise_induced_angina','oldpeak':'ST_depression',
        'slope':'slope_peak_exercise_ST','ca':'number_of_major_vessels'},
    inplace=True)



#  Update attribute values (features) for better readability & plotting
#=================================================
#  Update categorical values for better interpretation
#  (refer to dataset for values)
#  df     dataframe, for plotting
#  df          dataframe, for machine learning

# make copy of dataframe for plotting
#df = df.copy(deep=True)   # errors on Kaggle

#  Update categorical values for better interpretation
df['sex'] = df['sex'].map({0:'female', 1:'male'})
df['chest_pain_type'] = df['chest_pain_type'].map({
        0:'typical angina', 1:'atypical angina',
        2:'non-anginal',    3:'asymptomatic'})
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].map({
        0:'> 120 mg/dl', 1:'< 120 mg/dl'})
df['resting_electrocardiographic'] = df['resting_electrocardiographic'].map({
        0:'normal', 1:'ST-T wave abnormality', 2:'ventricular hypertrophy'})
df['exercise_induced_angina'] = df['exercise_induced_angina'].map({
        0:'no', 1:'yes'})
df['slope_peak_exercise_ST'] = df['slope_peak_exercise_ST'].map({
        0:'upsloping', 1:'flat', 2:'downsloping'})
df['thal'] = df['thal'].map({
        0:'normal 0',     1:'normal 1',
        2:'fixed defect', 3:'reversable defect'})
df['target'] = df['target'].map({0:'no disease', 1:'disease'})


df[['sex','target']].groupby(['sex']).count()
df[['sex','target']].groupby(['target']).count()


#  Define the data type
#=================================================
'''
  DATA TYPES:
      Categorical:
          Dichotomous:  sex
          Ordinal:      chest_pain_type, fasting_blood_sugar, resting_electrocardiographic, exercise_induced_angina, slope_peak_exercise_ST, number_of_major_vessels, thal
      Numerical:
          Continous:    age, resting_blood_pressure, cholestoral, maximum_heart_rate, ST_depression
'''


#  Easier way is to do this manually.  Doing it with a loop for larger datasets

#  Separate out Categorical and Numeric data
colCAT = []
colNUM = []
for i in df.columns:
    if (len(df[i].unique())) > 5:
        colNUM.append(i)
    else:
        colCAT.append(i)
    print('unique values:  {}\t{}'.format(len(df[i].unique()),i))

dataCAT = df[colCAT];dataCAT.info()   #  Categorical data, use histogram plot
colNUM.append('target')                    #  add target column to Numeric
dataNUM = df[colNUM];dataNUM.info()   #  Numeric data, use scatter plot

dataCAT.describe(include=['O'])
#####################################################




#####################################################
#  STEP 5:  Plot the Dataset
#####################################################
diseaseCAT    = df[(df['target'] == 'disease')]
no_diseaseCAT = df[(df['target'] == 'no disease')]

#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfRow = len(dataCAT.columns)
subNumOfCol = 3     # three columns: overall, no disease, disease
subPlotNum  = 1     # initialize plot number

fig = plt.figure(figsize=(16,60))
sns.set_palette("bright")

for i in colCAT:
    # overall
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('OVERALL - {}'.format(i), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.swarmplot(data=df, x=df[i],y=df.age,hue=df.target)
    subPlotNum = subPlotNum + 1
    # no_diseaseCAT
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('NO DISEASE, target = 0', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.swarmplot(data=no_diseaseCAT, x=no_diseaseCAT[i],y=no_diseaseCAT.age,color='darkorange')
    subPlotNum = subPlotNum + 1
    # diseaseCAT
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('DISEASE, target = 1', fontsize=14)
    plt.xlabel(i, fontsize=12)
    #sns.countplot(diseaseCAT[i], hue=df.sex)#,color='darkred')
    sns.swarmplot(data=diseaseCAT, x=diseaseCAT[i],y=diseaseCAT.age,color='blue')
    subPlotNum = subPlotNum + 1
plt.show()


'''
#  Plots: Overall, no disease and disease (side by side)
#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
#  assign CAT dataframe for "no disease" and "disease"
diseaseCAT    = dataCAT[(df['target'] == 'disease')]
no_diseaseCAT = dataCAT[(df['target'] == 'no disease')]

fig = plt.figure(figsize=(16,60))
sns.set_palette("bright")

for i in dataCAT.columns:
    # overall
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('Overall', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df[i], hue=df.sex)#,color='PiYG')
    subPlotNum = subPlotNum + 1
    # no_diseaseCAT
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('NO DISEASE, target = 0', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(no_diseaseCAT[i], hue=df.sex)#, color='darkgreen')
    subPlotNum = subPlotNum + 1
    # diseaseCAT
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('DISEASE, target = 1', fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(diseaseCAT[i], hue=df.sex)#,color='darkred')
    subPlotNum = subPlotNum + 1
    
plt.show()
'''


#  assign NUM dataframe for "no disease" and "disease"
no_diseaseNUM = dataNUM[(df['target'] == 'no disease')]
diseaseNUM    = dataNUM[(df['target'] == 'disease')]

#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfRow = len(dataNUM.columns)-1   #  x='age' in plots, drop column
subNumOfCol = 3     # three columns: overall, no disease, disease
subPlotNum  = 1     # initialize plot number

fig = plt.figure(figsize=(16,30))

for i in dataNUM.columns.drop(["age","target"]):
    # overall
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('OVERALL', fontsize=14)
    plt.xlabel(i, fontsize=12)
    #sns.scatterplot(data=df,x='age',y=df[i],hue='sex')
    sns.distplot(df[i],color='black')
    subPlotNum = subPlotNum + 1
    # no_diseaseNUM
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('NO DISEASE, target = 0', fontsize=14)
    plt.xlabel(i, fontsize=12)
    #sns.scatterplot(data=df,x='age',y=no_diseaseNUM[i],hue='sex')
    sns.distplot(no_diseaseNUM[i],color='darkorange')
    subPlotNum = subPlotNum + 1
    # diseaseNUM
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title('DISEASE, target = 1', fontsize=14)
    plt.xlabel(i, fontsize=12)
    #sns.scatterplot(data=df,x='age',y=diseaseNUM[i],hue='sex')
    sns.distplot(diseaseNUM[i],color='darkblue')
    subPlotNum = subPlotNum + 1

plt.show()
#####################################################


#####################################################
#  MODELING
#####################################################
df_backup = df.copy(deep=True)   # errors on Kaggle, make backup
df = df_backup.copy(deep=True)

#=================================================
#  One Hot Encoding
#=================================================
#  one hot encoding works on type 'object'
for i in colCAT:
    df[i] = df[i].astype(object)

df_OHE = df[colCAT]
df_OHE = pd.get_dummies(df_OHE)

#  add numeric columns to df_OHE dataframe
df_OHE = df_OHE.join(df[colNUM])

#  change target data to 0/1
df_OHE['target'] = df_OHE['target'].map({'no disease':0,'disease':1})
df_OHE = df_OHE.drop(['target_disease', 'target_no disease'], axis=1)
#=================================================


#=================================================
#  Normalize the data
#=================================================
norm = MinMaxScaler().fit_transform(df_OHE)
#norm = StandardScaler().fit_transform(df_OHE)
norm[0:10]

#  CREATE DATAFRAME WITH ONE HOT AND NORMALIZED DATA
df = pd.DataFrame(norm, index=df_OHE.index, columns=df_OHE.columns)

df.info()

#  Move target column to first column
'''
cols = list(df.columns)
cols.insert(-1, cols.pop(cols.index('target')))
df= df.reindex(columns = cols)
'''
#=================================================


#=================================================
#   CORRELATION
#=================================================
dataCorr = df.corr()

#  Correlation HEATMAP
plt.figure(figsize=(20,20))
plt.title('Heart Disease - CORRELATION, Overall', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')

#  Correlation TABLE
corrALL = dataCorr['target'].sort_values(ascending=False)
corrALL = corrALL.drop(['target'])
corrALL.to_frame()

#  Correlation BARPLOT
plt.figure(figsize=(16,16))
plt.title('Heart Disease - CORRELATION, Overall', fontsize=14)
ax = sns.barplot(y=corrALL.index,x=corrALL.values)
for p in ax.patches:
    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))
plt.show()


#  Correlation FEMALE - filter dataframe for male/female
#------------------------------------------
dataFemale = df[(df['sex_female'] > 0)]
dataFemaleCorr = dataFemale.drop(['sex_female','sex_male'], axis=1).corr()

#  HEATMAP
plt.figure(figsize=(10,10))
plt.title('correlation Heart Disease - FEMALE', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')

dataFemaleCorr = dataFemaleCorr['target'].sort_values(ascending=False)
dataFemaleCorr['number_of_major_vessels_4'] = 0  # -7.9e-17  all numbers will be exp if not set to 0
dataFemaleCorr.to_frame()
dataFemaleCorr = dataFemaleCorr.drop(['target'])  # for barplot

#  BARPLOT
plt.figure(figsize=(8,8))
plt.title('Heart Disease - CORRELATION, Female', fontsize=14)
ax = sns.barplot(y=dataFemaleCorr.index,x=dataFemaleCorr.values)
for p in ax.patches:
    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))
plt.show()


#  Correlation MALE - filter dataframe for male/female
#------------------------------------------
dataMale   = df[(df['sex_male'] > 0)]
dataMaleCorr = dataMale.drop(['sex_female','sex_male'], axis=1).corr()

#  HEATMAP
plt.figure(figsize=(10,10))
plt.title('correlation Heart Disease - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')

dataMaleCorr = dataMaleCorr['target'].sort_values(ascending=False)
dataMaleCorr.to_frame()
dataMaleCorr = dataMaleCorr.drop(['target'])  # for barplot

#  BARPLOT
plt.figure(figsize=(8,8))
plt.title('Heart Disease - CORRELATION, Male', fontsize=14)
ax = sns.barplot(y=dataMaleCorr.index,x=dataMaleCorr.values)
for p in ax.patches:
    ax.annotate("%.4f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()))
plt.show()


#  Correlation - sorted for both male/female
corrALL = pd.DataFrame(columns = ['attribute-MALE','correlation-m','attribute-FEMALE','correlation-f'])
corrALL['attribute-MALE']   = dataMaleCorr.index
corrALL['correlation-m'] = dataMaleCorr.values
corrALL['attribute-FEMALE'] = dataFemaleCorr.index
corrALL['correlation-f'] = dataFemaleCorr.values
print(corrALL)
#=================================================

#####################################################


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
X = df.drop(['target'], axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=seed)
print ('Train set:  ', X_train.shape,  y_train.shape)
print ('Test set:   ', X_test.shape,  y_test.shape)
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
#  5.5 Classification Model - Logistic Regression
#=================================================
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#  predict
y_predict = LR.predict(X_test)
y_predict[0:10]
#=================================================


#=================================================
##  6.1 Cross Validation Score<a id="eval_cv"></a>  
#=================================================
#from sklearn.model_selection import cross_val_score

print(cross_val_score(LR, X_train, y_train, cv=5, scoring='accuracy'))
print('Cross Validation Score (mean):  {:3.4%}'.format(cross_val_score(LR, X_train, y_train, cv=5, scoring='accuracy').mean()))
#=================================================


#=================================================
##  6.2 Accuracy Score<a id="eval_acc"></a>   
#=================================================
#from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_predict)
print('Accuracy Score:  {:3.4%}'.format(accuracy_score(y_test,y_predict)))

#---------------
names = []
modelAC = pd.DataFrame(columns=['model','AccuracyScore'])
countAC = 0

for name, model in models:
    names.append(name)
    modelPredict = model.predict(X_test)
    accu = accuracy_score(y_test,modelPredict)
    modelAC.loc[countAC]=[name,accu]
    countAC = countAC + 1

# sort on Accuracy Score
print("\nACCURACY SCORE:\n{}".format(modelAC.sort_values(['AccuracyScore'], ascending=False)))
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

#---------------
names = []
modelCM = pd.DataFrame(columns=['model','F1 Score'])
countCM = 0

for name, model in models:
    names.append(name)
    model.fit(X_train,y_train)
    modelPredict = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, modelPredict)
    f1score = f1_score(y_test, modelPredict)
    modelCM.loc[countCM]=[name,f1score]
    sns.heatmap(conf_matrix, annot=True,cmap='Blues',annot_kws={"size": 30})
    plt.title("{} - Confusion Matrix, F1 Score: {:3.4%}".format(name,f1score))
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.show()
    countCM = countCM + 1

# sort on F1 Score
print("\nF1 SCORE:\n{}".format(modelCM.sort_values(['F1 Score'], ascending=False)))
#=================================================


#=================================================
##  6.5 Receiver Operating Characteristics (ROC) Curve<a id="eval_roc"></a>   
#=================================================
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score

LR.probability = True   # need for predict_proba to work
LR.fit(X_train,y_train)
y_predita = LR.predict_proba(X_test)
y_predita = y_predita[:,1]   # positive values only
    
ROC_AUC = roc_auc_score(y_test, y_predita)
fpr, tpr, thresholds = roc_curve(y_test, y_predita)

plt.plot([0,1],[0,1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title("ROC Curve, ROC_AUC Score: {:3.4%}".format(ROC_AUC))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


#---------------
names = []
modelRO = pd.DataFrame(columns=['model','AUC'])
countRO = 0

for name, model in models:
    names.append(name)
    model.probability = True   # need for predict_proba to work
    model.fit(X_train,y_train)
    modelPredict = model.predict_proba(X_test)
    modelPredict = modelPredict[:,1]   # positive values only
    ROC_AUC = roc_auc_score(y_test, modelPredict)
    fpr, tpr, thresholds = roc_curve(y_test, modelPredict)
    modelRO.loc[countRO]=[name,ROC_AUC]
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title("{} - ROC Curve, ROC_AUC Score: {:3.4%}".format(name,ROC_AUC))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    countRO = countRO + 1

# sort on F1 Score
print("\nAUC:\n{}".format(modelRO.sort_values(['AUC'], ascending=False)))
#=================================================


#=================================================
##  6.6 Classification Report<a id="eval_class"></a>   
#=================================================
#from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))

#---------------
names = []
countCR = 0

for name, model in models:
    names.append(name)
    model.fit(X_train,y_train)
    modelPredict = model.predict(X_test)
    print("\n\t==========  {}  ==========\n{}".format(name,classification_report(y_test,modelPredict)))
    countCR = countCR + 1
#=================================================


#=================================================
##  6.7 Log Loss<a id="eval_log"></a>   
#=================================================
#from sklearn.metrics import log_loss

#  predict_proba returns estimates for all classes
y_predict_prob = LR.predict_proba(X_test)
print(y_predict_prob[0:5])

print("\nLog Loss:  {:3.4}".format(log_loss(y_test, y_predict_prob)))

#---------------
names = []
modelLL = pd.DataFrame(columns=['model','Log Loss'])
countLL = 0

for name, model in models:
    names.append(name)
    #modelPredict = model.predict_proba(X_test)  #  predict_proba
    modelPredict = model.predict(X_test)
    logLoss = log_loss(y_test, modelPredict)
    modelLL.loc[countLL]=[name,logLoss]
    countLL = countLL + 1

# sort on Log Loss
print("\LOG LOSS:\n{}".format(modelLL.sort_values(['Log Loss'], ascending=False)))
#=================================================





#=================================================
#  COMBINE ALL EVALUATIONS
#  Evaluate the Models
#   - Cross-Validation
#   - Accuracy Score
#   - Confusion Matrix
#   - F1 Score
#   - ROC AUC, ROC Curve (Receiver Operating Characteristics, area under curve)
#   - Classification Report
#   - Log Loss
#=================================================
results = []
names = []
modelCV = pd.DataFrame(columns=['model','CV-mean','CV-std','AccuracyScore','F1Score','ROC AUC','Log Loss'])
countCV = 0

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    model.probability = True   # need for predict_proba to work
    model.fit(X_train,y_train)
    modelPredict  = model.predict(X_test)        #  predict
    modelPredicta = model.predict_proba(X_test)  #  predict_proba
    modelPredicta = modelPredicta[:,1]           # positive values only
    model.fit(X_train,y_train)
    accu = accuracy_score(y_test,modelPredict)          #  accuracy score
    f1Score = f1_score(y_test,modelPredict)             #  F1 score
    conf_matrix = confusion_matrix(y_test, y_predict)   #  confusion matrix
    class_report = classification_report(y_test,modelPredict)  # class report
    logLoss = log_loss(y_test, modelPredicta)            # log loss
    ROC_AUC = roc_auc_score(y_test, modelPredicta)           # area under curve
    fpr, tpr, thresholds = roc_curve(y_test, modelPredicta)    #  ROC curve
    print("\n\n\t============  {}  ============".format(name))
    print("Cross Validation:\t{:3.4%}  {:3.4}".format(cv_results.mean(),cv_results.std()))
    print("Accuracy Score:\t\t{:3.4%}".format(accu))
    print("F1 Score:\t\t{:3.4%}".format(accu))
    print("ROC AUC:\t\t{:3.4%}".format(accu))
    print("Log Loss:\t\t{:3.4}".format(logLoss))
    print("\n\t-----  {}  Classification Report-----\n{}".format(name,class_report))
    #  Plot Confusion Matrix
    sns.heatmap(conf_matrix, annot=True,cmap='Blues',annot_kws={"size": 30})
    plt.title("{} - Confusion Matrix, F1 Score: {:3.4%}".format(name,f1Score))
    plt.show()
    #  Plot ROC Curve
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title("{} - ROC Curve, ROC_AUC Score: {:3.4%}".format(name,ROC_AUC))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    modelCV.loc[countCV]=[name,cv_results.mean(),cv_results.std(),accu,f1Score,ROC_AUC,logLoss]
    countCV = countCV + 1

#  Dataframe modelCV
print("\nsort on CV-mean:\n{}".format(modelCV.sort_values(['CV-mean'], ascending=False)))
print("\nsort on AccuracyScore:\n{}".format(modelCV.sort_values(['AccuracyScore'], ascending=False)))
print("\nsort on F1Score:\n{}".format(modelCV.sort_values(['F1Score'], ascending=False)))
print("\nsort on F1Score:\n{}".format(modelCV.sort_values(['F1Score'], ascending=False)))
print("\nsort on Log Loss:\n{}".format(modelCV.sort_values(['Log Loss'], ascending=False)))
#=================================================
#####################################################





