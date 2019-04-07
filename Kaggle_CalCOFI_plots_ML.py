# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:30:01 2019

@author: Asim Islam

---------------------------------------------------------------------------
data SET:   CalCOFI  (https://www.kaggle.com/sohier/calcofi) ---------------------------------------------------------------------------
  - Import Libraries and Llad the Dataset
  - Data Cleaning
  - Exploratory Data Analysis (EDA)
"""

#===============================================
#  STEP 1:  Import Libraries
#===============================================
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 80, 'display.max_rows', 50)
#===============================================

#===============================================
#  STEP 2:  Load the Dataset
#===============================================
cofi=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/CalCOFI/bottle.csv')
#===============================================


############################################################
#   DATA CLEANING
#       1) Extract Year and Month from Depth_ID field 
#       2) drop reference columns
#       3) check for NULL and DUPLICATES
#       4) Determine if NULLs should be filled with NA or mean
#       5) visualize datapoint collection over the years
############################################################

#===============================================
#  1) Extract Year and Month from Depth_ID field 
#        Depth_ID = [Century]-[YY][MM][ShipCode]-etc
#        19-4903CR-HY-060-0930-05400560-0020A-7
#===============================================
# year = x.split('-')[0]+x.split('-')[1][0:2]
# month = x.split('-')[1][2:4]
cofi['Year'] = (cofi['Depth_ID'].str.split('-', expand=True)[0] + \
                cofi['Depth_ID'].str.split('-', expand=True)[1]). \
                map(lambda x: str(x)[:4])
cofi['Month'] = (cofi['Depth_ID'].str.split('-', expand=True)[1]). \
                 map(lambda x: str(x)[2:4])
                 
cofi[['Depth_ID','Year','Month']].head(10)
#===============================================


#===============================================
#  2)  Drop Reference Columns
#       o Cst_Cnt   Auto-numbered Cast Count
#       o Btl_Cnt   Auto-numbered Bottle count
#       o Sta_ID    CalCOFI Line and Station
#       o Depth_ID  [Century]-[YY][MM][ShipCode]-deleted
#===============================================
cofi = cofi.drop(['Cst_Cnt','Btl_Cnt','Sta_ID','Depth_ID'], axis=1)

cofi.info()
#===============================================


#===============================================
#  3)  Check for NULLs and DUPLICATED
#       - drop DUPLICATED rows
#       - drop columns with more than 20% missing data
#       - PLOTS:  normalize, then fill with NULL and fill with MEAN
#===============================================
#  drop DUPLICATES rows
cofi.duplicated().sum()               #  count of duplicate values
cofi.drop_duplicates(inplace = True)
cofi.duplicated().sum()               #  count of duplicate values

# drop columns with more than 70% data missing
a = cofi.isna().sum()
b = pd.DataFrame({'coln':a.index, 'NaN_count':a.values})
percentNA = 0.7

nullPercent = (1-percentNA) * cofi.shape[0]
for i in a.index:
    c = (a[i] > nullPercent)
    if (c == True):
        cofi.drop(i, axis=1, inplace=True)
        #print("drop ", i, a[i])
        
print("DUPLICATES:  ", cofi.duplicated().sum(),"\n")
cofi.info()
#===============================================


#===============================================
#   4) Determine if NULLs should be filled with NA or mean
#      - PLOTS:  normalize, then fill with NULL and fill with MEAN
#      - Determine from plots if NULLs should be NaN or Mean
#           o normalized plots should be smooth
#           o use whichever plot matches the normalized plot the best
#           o fill mean if the plot matches normalized data
#           o fill na if mean plot does not match normalized
#===============================================
#  find columns with NULL values
colnNULL = cofi.isnull().sum()        #  count of null values
cofiNULL = []

for i in cofi.isnull().sum().index:
    if (colnNULL[i] != 0):
        cofiNULL.append(i)          #  columns with NULL values
    print(i,colnNULL[i])


#  PLOTS:  normalize, then fill with NULL and fill with MEAN
#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfCol = 6                 # each attribute has 3 columns
subNumOfRow = len(cofiNULL)/2   # two attributes per row
subPlotNum  = 1                 # initialize plot number

fig = plt.figure(figsize=(16,60))

for i in cofiNULL:
    # normalized
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("NORMALIZED", fontsize=10)
    plt.xlabel(i, fontsize=12)
    normPlot = stats.boxcox(cofi[cofi[i] > 0][i])[0]  # normalize
    sns.distplot(normPlot, color='black')
    subPlotNum = subPlotNum + 1
    # fill NA
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("fill=NA", fontsize=10)
    plt.xlabel(i, fontsize=12)
    fillNaPlot = cofi[i].dropna()   # fill na
    sns.distplot(fillNaPlot, color='green')
    subPlotNum = subPlotNum + 1
    # fill MEAN
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("fill=MEAN", fontsize=10)
    plt.xlabel(i, fontsize=12)
    fillMePlot = cofi[i].fillna(cofi[i].mean())  # fill mean
    sns.distplot(fillMePlot, color='blue')
    subPlotNum = subPlotNum + 1
    
plt.show()


#  Visually inspect the normalized, fill na and fill mean plots
#  columns to fill with NA
cofiFillNa = ['O2ml_L']

#  columns to fill with mean value
cofiFillMe = ['T_degC', 'Salnty', 'STheta', 'O2Sat', 
              'Oxy_µmol/Kg', 'T_prec', 'S_prec', 
              'P_qual', 'Chlqua', 'Phaqua', 'NH3q', 
              'C14A1q', 'C14A2q', 'DarkAq', 'MeanAq',
              'R_TEMP', 'R_POTEMP', 'R_SALINITY', 
              'R_SIGMA', 'R_SVA', 'R_DYNHT', 'R_O2', 
              'R_O2Sat']


for i in cofiFillNa:
    cofi[i].dropna(inplace=True)   
#  dropna did not work - may need to delete column:  del cofi['O2ml_L']

for i in cofiFillMe:
    cofi[i].fillna(cofi[i].mean(), inplace=True)  # fill mean

cofi.isnull().sum()        #  count of null values
#===============================================


############################################################
#   Exploratory Data Analysis (EDA)
#       1)  Salinity over the years and seasonal
#       2)  Correlation - heatmap
#       3)  Plot high-correlation features
############################################################
#===============================================
#  1) visualize datapoint collection over the years
#       - Salinity over years
#       - Salinity over months
#  NOTE: use scatterplot.  All other plots take too long to draw
#===============================================
#  Salinity distribution
plt.figure(figsize=(8,6))
plt.xlim([32, 36])
plt.title('Salinity Distribution', fontsize=14)
sns.distplot(cofi['Salnty'])

#  Yearly change in Salinity
fig = plt.figure(figsize=(12,6))
fig.autofmt_xdate()
fig.add_subplot(121)
plt.title('Yearly change in Salinity', fontsize=14)
sns.scatterplot(data=cofi, x='Year', y='Salnty')
#  Seasonal change in Salinity
fig.add_subplot(122)
plt.title('Seasonal change in Salinity', fontsize=14)
sns.scatterplot(data=cofi, x='Month', y='Salnty')
plt.show()
#===============================================


#===============================================
#  2) Correlation - heatmap
#===============================================
cofiCorr = cofi.corr()
plt.figure(figsize=(16,16))
plt.title('Salinity Correlation Heatmap', fontsize=14)
sns.heatmap(cofiCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')
#===============================================


#===============================================
#  3) Plot high-correlation features
#===============================================
cofiSalnCorr = cofiCorr['Salnty'].sort_values(ascending=False)
cofiSalnCorr.head(10)   # top 10 positive correlations
cofiSalnCorr.tail(10)   # top 10 negative correlations

#  Plot top 2 high +ve & -ve Salinity correlation result
#  faster then pairplot
subNumOfRow = 2
subNumOfCol = 2
subPlotNum  = 1     # initialize plot number

fig = plt.figure(figsize=(10,10))

for i in ['R_SALINITY','R_DYNHT','Oxy_µmol/Kg','R_O2']:
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.xlabel(i, fontsize=10)
    plt.title('Correlation - Salinity', fontsize=12)
    sns.scatterplot(data=cofi,x=cofi[i],y='Salnty', color='Green')
    subPlotNum = subPlotNum + 1

plt.show()


#cofiPP = ['R_SALINITY','R_DYNHT','Oxy_µmol/Kg','R_O2']
#cofiPP1 = pd.DataFrame(cofi[['Salnty', 'R_SALINITY', 'NO3uM', 'R_NO3', #'O2ml_L', 'R_O2', 'Oxy_µmol/Kg']])
#sns.pairplot(cofiPP1, x_vars=cofiPP, y_vars='Salnty')
#===============================================




