# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:21:45 2019

@author: ACER

ML Tutorial -- subplots
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
#  Import Libraries
#####################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/heart.csv')
#####################################################


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

age         303 non-null int64
sex         303 non-null int64
cp          303 non-null int64
trestbps    303 non-null int64
chol        303 non-null int64
fbs         303 non-null int64
restecg     303 non-null int64
thalach     303 non-null int64
exang       303 non-null int64
oldpeak     303 non-null float64
slope       303 non-null int64
ca          303 non-null int64
thal        303 non-null int64
target      303 non-null int64

    Nominal -   variables that have two or more categories, but 
    which do not have an intrinsic order
    Dichotomous - Nominal variable with only two categories
    Ordinal -   variables that have two or more categories just 
    like nominal variables. Only the categories can also be 
    ordered or ranked
    
#  CATEGORICAL:
   'sex','cp','fbs','restecg','exang','slope','ca','thal','target,'
#  NUMERIC:
#  'age','trestbps','chol','thalach','oldpeak'
"""

"""
TUTORIAL:  SUPLOTS
Subplots are an excellent tools for data visualization.  Data can be plotted side-by-side for comparison, multiple plots can show inter-relationships and subplots provide some more control over what needs to be plotted than FactorGrid and PairPlots.

SUB-PLOT COMPONENTS:
figure(figsize=(8,8))       - total plot area size in (x,y) inches
add_subplot(a,b,c)          - a=NumOfRow, b=NumOfCol, c=subCnt
    add_subplot(121) and add_subplot(1,2,1) are the same
    plt.show()              - plot the sub=plots
    
    countplot
    scatterplot
    boxplot
    lineplot
    heatmap
"""

#--------------------------------------
#  1x2  - 2 plots
#--------------------------------------
fig = plt.figure(figsize=(8,4))

#  subplot #1
fig.add_subplot(121)
plt.title('subplot(121)', fontsize=14)
sns.countplot(data=df, x='cp')

#  subplot #2
fig.add_subplot(122)
plt.title('subplot(122)', fontsize=14)
sns.scatterplot(data=df,x='age',y='chol',hue='sex')

plt.show()
#--------------------------------------


#--------------------------------------
#  2x3  - 5 plots
#   'sex','cp','fbs','restecg','exang','slope','ca','thal','target,'
#  NUMERIC:
#  'age','trestbps','chol','thalach','oldpeak'
#--------------------------------------
fig = plt.figure(figsize=(14,12))

#  subplot #1
fig.add_subplot(231)
plt.title('subplot(231) - title', fontsize=14)
sns.countplot(data=df, x='cp',hue='sex')

#  subplot #2
fig.add_subplot(2,3,2)
plt.title('subplot(2,3,2)', fontsize=14)
sns.scatterplot(data=df,x='age',y='chol',hue='sex')

#  subplot #3
fig.add_subplot(233)
plt.title('subplot(233)', fontsize=14)
sns.lineplot(data=df, x=df['age'],y=df['oldpeak'])

#  subplot #4
fig.add_subplot(2,3,4)
plt.title('subplot(2,3,4)', fontsize=14)
sns.boxplot(data=df[['chol','trestbps','thalach']])

#  subplot #5
fig.add_subplot(235)
plt.title('subplot(235)', fontsize=14)
sns.countplot(data=df, x='slope',hue='sex')

plt.show()
#--------------------------------------



#--------------------------------------
#  multiple sub-plots with FOR loop
#  plot each feature for overall, no disease, disease
#--------------------------------------
#  Plots: Overall, no disease and disease (side by side)
df2 = df[['sex','cp','slope','ca']]        # select few attributes
df_target_0 = df[(df['target'] == 0)]      # "no disease" data
df_target_1    = df[(df['target'] == 1)]   # "disease" data


#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
rowCnt = len(df2.columns)
colCnt = 3  # three columns: overall, no disease, disease
subCnt = 1  # initialize plot number

fig = plt.figure(figsize=(12,30))

for i in df2.columns:
    # OVERALL plots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('OVERALL (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df[i], hue=df.sex)
    subCnt = subCnt + 1

    # NO DISEASE PLOTS
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('NO DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df_target_0[i], hue=df.sex)
    subCnt = subCnt + 1

    # PLOTS
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.title('DISEASE (row{},col{},#{})'.format(rowCnt, colCnt, subCnt), fontsize=14)
    plt.xlabel(i, fontsize=12)
    sns.countplot(df_target_1[i], hue=df.sex)
    subCnt = subCnt + 1

plt.show()
#--------------------------------------


#--------------------------------------
#  HEATMAP subplots
#--------------------------------------
dfFemale = df2[(df2['sex'] == 1)]                       # female
dfFemaleCorr = dfFemale.drop(["sex"], axis=1).corr()    # female corr
dfMale   = df2[(df2['sex'] == 0)]                       # male
dfMaleCorr = dfMale.drop(["sex"], axis=1).corr()        # male corr


fig = plt.figure(figsize=(12,6))
#  heatmap - female
fig.add_subplot(121)
plt.title('correlation Heart Disease - FEMALE', fontsize=14)
sns.heatmap(dfFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
#  heatmap - male
fig.add_subplot(122)
plt.title('correlation Heart Disease - MALE', fontsize=14)
sns.heatmap(dfMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')

plt.show()
#--------------------------------------



