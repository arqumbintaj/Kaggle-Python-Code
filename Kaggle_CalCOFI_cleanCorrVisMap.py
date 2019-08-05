# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:30:01 2019

@author: Asim Islam

---------------------------------------------------------------------------
data SET:   Caldf  (https://www.kaggle.com/sohier/caldf) ---------------------------------------------------------------------------
  - Import Libraries and Llad the Dataset
  - Data Cleaning
  - Exploratory Data Analysis (EDA)
  - Plot Locations Data on World Map
"""

#===============================================
#  Import Libraries
#===============================================
import pandas as pd
import matplotlib.pyplot as plt
#from scipy import stats
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler

#  maps
import folium
from folium.plugins import MarkerCluster

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
sns.palplot(sns.color_palette("Blues",5))
sns.palplot(sns.color_palette("BrBG", 4))
sns.palplot(sns.color_palette("bright"))
plt.show()
#===============================================

#===============================================
#  Load the Dataset
#===============================================
df= pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/CalCOFI/bottle.csv')

#  bottle.csv contains information on ocean conditions
#  cast.csv   contains information on collecting stations
#===============================================

###########################
#df = df.head(50000)  #  make dataframe smaller till code is done
###########################


#===============================================
#  CHECK FOR NULLs and DUPLICATES
#===============================================
#plt.title('DATA - "nulls\"')
#sns.heatmap(df.isnull())

nulls = df.isnull().sum().sort_values(ascending = False)
prcet = round(nulls/len(df)*100,2)

df_null = pd.DataFrame(columns =  ['Attr','Total','Percent'])
df_null.Attr  = nulls.index
df_null.Total = nulls.values
df_null.Percent = prcet.values
print(df_null)


#  drop columns with more then 30% nulls
for i in df_null.Attr[df_null['Percent'] > 30]:
    df = df.drop([i], axis=1)
    print(df.shape,i)


#  Fill all NaN with MODE of that attribute
for i in df.columns:
    print(i,df[i].isnull().sum())
    if df[i].isnull().sum() > 0:
        df[i].fillna(df[i].mode().mean(), inplace=True)
        print('filled: {}\tmode: {}\tmean: {}'.format(i,df[i].mode(),df[i].mode().mean()))

#plt.title('DATA - "nulls\"')
#sns.heatmap(df.isnull())


#  check for duplicates
df.duplicated().sum()
#===============================================



#===============================================
#  Extract Year and Month from Depth_ID field 
#        Depth_ID = [Century]-[YY][MM][ShipCode]-etc
#        19-4903CR-HY-060-0930-05400560-0020A-7
#===============================================
# year = x.split('-')[0]+x.split('-')[1][0:2]
# month = x.split('-')[1][2:4]
df['Year'] = (df['Depth_ID'].str.split('-', expand=True)[0] + \
                df['Depth_ID'].str.split('-', expand=True)[1]). \
                map(lambda x: str(x)[:4])
df['Month'] = (df['Depth_ID'].str.split('-', expand=True)[1]). \
                 map(lambda x: str(x)[2:4])
                 
df[['Depth_ID','Year','Month']].head(10)
#===============================================


#===============================================
#  Drop columns that cannot be normalized
#===============================================
drop_cols = ['Cst_Cnt', 'Btl_Cnt', 'Sta_ID', 'Depth_ID', 'Depthm','Year','Month']
df_norm = df.drop(drop_cols, axis=1)  #  data for normalization
df_scale = df_norm.copy(deep=True)    #  backup data
#===============================================


#===============================================
#  Normalize the data
#===============================================
df_scale = StandardScaler().fit_transform(df_scale)

#  create dataframe
df_norm = pd.DataFrame(df_scale, index=df_norm.index, columns=df_norm.columns)
df_norm.isnull().sum()
#===============================================


#===============================================
#  2) Correlation - heatmap
#===============================================
df_norm.corr()

#  Drop columns with mode = "0.0".  No impact on correlation
for i in df_norm.columns.tolist():
    if (df_norm[i].mode()[0] == 0.0):
        #print(' - ',i,df_norm[i].mode()[0])
        print(' - ',i,df_norm[i].mode())
        df_norm = df_norm.drop(i,axis=1)
    else:
        #print(i,df_norm[i].mode()[0])
        print(i,df_norm[i].mode())


#  Create correlation dataframe
df_corr = pd.DataFrame(columns=['Attributes','Correlation'])
df_corr.Attributes = df_norm.corr()['Salnty'].sort_values(ascending=False).index
df_corr.Correlation = df_norm.corr()['Salnty'].sort_values(ascending=False).values



'''
#  correlation heatmap
plt.figure(figsize=(16,16))
plt.title('Salinity Correlation Heatmap', fontsize=14)
sns.heatmap(df_norm, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')


sns.lmplot(x="T_degC",y="Salnty",data=df,order=1, ci=None,scatter_kws={'alpha':0.15});
sns.regplot(x="T_degC",y="Salnty",data=df,order=1, ci=None,scatter_kws={'alpha':0.15});
sns.residplot(df.T_degC,df.Salnty, order=2, lowess=True)
plt.xlim(-5,5)
sns.distplot(df_norm.Salnty)
sns.distplot(df_norm.T_degC)
plt.show()
sns.distplot(df_norm['Oxy_µmol/Kg'])

Salnty         1.000000
R_SALINITY     1.000000
STheta         0.828479
R_SIGMA        0.824973
R_DYNHT        0.794913
R_Depth        0.726874
R_PRES         0.726216
S_prec         0.120530
RecInd         0.118570
T_prec         0.062595
R_TEMP        -0.662933
T_degC        -0.662933
R_O2Sat       -0.812163
O2Sat         -0.814513
R_SVA         -0.821338
O2ml_L        -0.827010
R_O2          -0.827010
Oxy_µmol/Kg   -0.836961
'''
#===============================================



############################################################
#  DATA VISUALIZATION
############################################################

#===============================================
#       - Salinity over years
#       - Salinity over months
#===============================================
#  Salinity distribution
plt.figure(figsize=(8,6))
plt.xlim([32, 36])
plt.title('Salinity Distribution', fontsize=14)
sns.distplot(df['Salnty'])

#  Yearly change in Salinity
fig = plt.figure(figsize=(12,6))
fig.autofmt_xdate()
fig.add_subplot(121)
plt.title('Yearly change in Salinity', fontsize=14)
sns.scatterplot(data=df, x='Year', y='Salnty')

#  Seasonal change in Salinity
fig.add_subplot(122)
plt.title('Seasonal change in Salinity', fontsize=14)
sns.scatterplot(data=df, x='Month', y='Salnty')
plt.show()
#===============================================



#===============================================
#  SAMPLE THE DATA
#  reduce the number of points for plotting
#  results in much cleaner plots
#===============================================
#get the top 5 and bottom 5 high-correlation attributes
df_high = df_corr[2:7]
df_high = df_high.append(df_corr[-5:])


#  take a sample of "df" features
#df_sample  = df.sample(n=int(round(len(df)*.002,0)), random_state=0)
#df_sampleN = df_norm.sample(n=int(round(len(df)*.002,0)), random_state=0)
df_sample = df_norm.sample(n=int(round(len(df)*.002,0)), random_state=0)
#===============================================


#===============================================
#  PLOT SELECTED COLUMNS IN CORR  - GOOD
#===============================================
#  PLOT THESE COLUMNS
plot_attr = ['R_DYNHT', 'R_SIGMA', 'R_Depth', 'RecInd', 'NH3q',  'T_prec', 'T_degC', 'R_POTEMP', 'O2ml_L']

for i in plot_attr:
    if plot_attr[0] == i:
        df_plot = df_corr[df_corr.Attributes == i]
    else:
        df_plot = df_plot.append(df_corr[df_corr.Attributes == i])

#  SUBPLOTS
fig = plt.figure(figsize=(14,60))
col = 3
row  = int(len(df_corr.Attributes)/col)
count = 1

for i, j in zip(df_plot.Attributes,df_plot.Correlation):
    fig.add_subplot(row, col, count)
    plt.title('Salinity vs {} (corr = {:.4})\nnormalized distribution'.format(i,j))
    plt.xlim(-4,4)
    sns.distplot(df_sample.Salnty)#, color='blue')
    sns.distplot(df_sample[i])#,color='green')
    count = count + 1

plt.show()


#  Plot high correlation attributes - PLOTS #3  -  GOOD
fig = plt.figure(figsize=(14,60))
col = 3
row  = int(len(df_corr.Attributes)/col)
count = 1

for i, j in zip(df_plot.Attributes,df_plot.Correlation):
    fig.add_subplot(row, col, count)
    plt.title('Salinity vs {} (corr = {:.4f})\nnormalized distribution'.format(i,j))
    #plt.xlim(-4,4)
    sns.regplot(x=df_sample[i],y="Salnty",data=df_sample,order=2, scatter_kws={'alpha':0.25},color='green');
    count = count + 1

plt.show()
#===============================================


#===============================================
#  PLOT ALL COLUMNS IN CORR  - not good
#===============================================
#  Plot high correlation attributes - PLOTS #1  - CRAP
fig = plt.figure(figsize=(12,60))
plotNum  = 1     # initialize plot number

#for i in df_high.columns.drop(['Salnty','R_SALINITY']):
for i, j in zip(df_high.Attributes,df_high.Correlation):
#for i in df_high.Attributes:
    fig.add_subplot(8,2,plotNum)
    plt.title('Salinity vs {} (corr = {:.4f})'.format(i,j), fontsize=14)
    plt.grid()
    sns.scatterplot(x=df_sample[i], y=df_sample.Salnty, color='green')
    plotNum = plotNum + 1

plt.show()


#  Plot high correlation attributes - PLOTS #2  -  GOOD
for i, j in zip(df_high.Attributes,df_high.Correlation):
    plt.title('Salinity vs {} (corr = {:.4})\nnormalized distribution'.format(i,j), fontsize=14)
    plt.xlim(-4,4)
    sns.distplot(df_sample.Salnty)#, color='blue')
    sns.distplot(df_sample[i])#,color='green')
    plt.show()


#  Plot high correlation attributes - PLOTS #3  -  GOOD
for i, j in zip(df_high.Attributes,df_high.Correlation):
    plt.title('Salinity vs {} (corr = {:.4f})\nnormalized distribution'.format(i,j), fontsize=14)
    #plt.xlim(-4,4)
    sns.regplot(x=df_sample[i],y="Salnty",data=df_sample,order=2, scatter_kws={'alpha':0.25},color='green');
    plt.show()
#===============================================





#===============================================






############################################################
#   Plot Locations Data on World Map
#       1)  Import "folium"
#       2)  Load Caldf data long/lat and dates
#       3)  Plot location points on world map
#
#  data SET:   caldf  (https://www.kaggle.com/sohier/caldf)
############################################################
#===============================================
#  Load the Dataset
#===============================================
dfLOC=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/CalCOFI/cast.csv')
#===============================================


############################################################
#   Plot Locations Data on World Map
#
#   GEO LOCATION
#       Decimal Degrees = degrees + (minutes/60) + (seconds/3600)
#       cast.csv:  Lat_Dec, Lon_Dec and Date
############################################################
dfLOC = dfLOC[['Lat_Dec', 'Lon_Dec','Date']]
dfLOC = dfLOC.tail(1000)
dfLOC = dfLOC.reset_index(drop=True)  # reset index after tail

#  Load empty map
salinity_map = folium.Map(location=[dfLOC.Lat_Dec.mean(),dfLOC.Lon_Dec.mean()], zoom_start=6)

marker_cluster = MarkerCluster().add_to(salinity_map)

for i in range(len(dfLOC)):
    folium.Marker(location=[dfLOC.Lat_Dec[i],dfLOC.Lon_Dec[i]],
            popup = (dfLOC.Date[i]),         # dates in popups
            icon = folium.Icon(color='green')  # green popup icon
    ).add_to(marker_cluster)

salinity_map.add_child(marker_cluster)
salinity_map.save("C:\\Users\ACER\Desktop\\salinity_mapFolium.html")
salinity_map         #  display map
#####################################################################










#/////////////////////////////////////////

############################################################
#   DATA CLEANING
#       1) Extract Year and Month from Depth_ID field 
#       2) drop reference columns
#       3) check for NULL and DUPLICATES
#       4) Determine if NULLs should be filled with NA or mean
#       5) visualize datapoint collection over the years
############################################################




'''
#===============================================
#  2)  Drop Reference Columns
#       o Cst_Cnt   Auto-numbered Cast Count
#       o Btl_Cnt   Auto-numbered Bottle count
#       o Sta_ID    Caldf Line and Station
#       o Depth_ID  [Century]-[YY][MM][ShipCode]-deleted
#===============================================
df = df.drop(['Cst_Cnt','Btl_Cnt','Sta_ID','Depth_ID'], axis=1)

df.info()
#===============================================


#===============================================
#  3)  Check for NULLs and DUPLICATED
#       - drop DUPLICATED rows
#       - drop columns with more than 20% missing data
#       - PLOTS:  normalize, then fill with NULL and fill with MEAN
#===============================================
#  drop DUPLICATES rows
df.duplicated().sum()               #  count of duplicate values
df.drop_duplicates(inplace = True)
df.duplicated().sum()               #  count of duplicate values

# drop columns with more than 70% data missing
a = df.isna().sum()
b = pd.DataFrame({'coln':a.index, 'NaN_count':a.values})
percentNA = 0.7

nullPercent = (1-percentNA) * df.shape[0]
for i in a.index:
    c = (a[i] > nullPercent)
    if (c == True):
        df.drop(i, axis=1, inplace=True)
        #print("drop ", i, a[i])
        
print("DUPLICATES:  ", df.duplicated().sum(),"\n")
df.info()
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
colnNULL = df.isnull().sum()        #  count of null values
dfNULL = []

for i in df.isnull().sum().index:
    if (colnNULL[i] != 0):
        dfNULL.append(i)          #  columns with NULL values
    print(i,colnNULL[i])


#  PLOTS:  normalize, then fill with NULL and fill with MEAN
#  fig.add_subplot([# of rows] by [# of columns] by [plot#])
subNumOfCol = 6                 # each attribute has 3 columns
subNumOfRow = len(dfNULL)/2   # two attributes per row
subPlotNum  = 1                 # initialize plot number

fig = plt.figure(figsize=(16,60))

for i in dfNULL:
    # normalized
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("NORMALIZED", fontsize=10)
    plt.xlabel(i, fontsize=12)
    normPlot = stats.boxcox(df[df[i] > 0][i])[0]  # normalize
    sns.distplot(normPlot, color='black')
    subPlotNum = subPlotNum + 1
    # fill NA
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("fill=NA", fontsize=10)
    plt.xlabel(i, fontsize=12)
    fillNaPlot = df[i].dropna()   # fill na
    sns.distplot(fillNaPlot, color='green')
    subPlotNum = subPlotNum + 1
    # fill MEAN
    fig.add_subplot(subNumOfRow, subNumOfCol, subPlotNum)
    plt.title("fill=MEAN", fontsize=10)
    plt.xlabel(i, fontsize=12)
    fillMePlot = df[i].fillna(df[i].mean())  # fill mean
    sns.distplot(fillMePlot, color='blue')
    subPlotNum = subPlotNum + 1
    
plt.show()


#  Visually inspect the normalized, fill na and fill mean plots
#  columns to fill with NA
dfFillNa = ['O2ml_L']

#  columns to fill with mean value
dfFillMe = ['T_degC', 'Salnty', 'STheta', 'O2Sat', 
              'Oxy_µmol/Kg', 'T_prec', 'S_prec', 
              'P_qual', 'Chlqua', 'Phaqua', 'NH3q', 
              'C14A1q', 'C14A2q', 'DarkAq', 'MeanAq',
              'R_TEMP', 'R_POTEMP', 'R_SALINITY', 
              'R_SIGMA', 'R_SVA', 'R_DYNHT', 'R_O2', 
              'R_O2Sat']


for i in dfFillNa:
    df[i].dropna(inplace=True)   
#  dropna did not work - may need to delete column:  del df['O2ml_L']

for i in dfFillMe:
    df[i].fillna(df[i].mean(), inplace=True)  # fill mean

df.isnull().sum()        #  count of null values
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
sns.distplot(df['Salnty'])

#  Yearly change in Salinity
fig = plt.figure(figsize=(12,6))
fig.autofmt_xdate()
fig.add_subplot(121)
plt.title('Yearly change in Salinity', fontsize=14)
sns.scatterplot(data=df, x='Year', y='Salnty')
#  Seasonal change in Salinity
fig.add_subplot(122)
plt.title('Seasonal change in Salinity', fontsize=14)
sns.scatterplot(data=df, x='Month', y='Salnty')
plt.show()
#===============================================


#===============================================
#  2) Correlation - heatmap
#===============================================
dfCorr = df.corr()
plt.figure(figsize=(16,16))
plt.title('Salinity Correlation Heatmap', fontsize=14)
sns.heatmap(dfCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')
#===============================================


#===============================================
#  3) Plot high-correlation features
#===============================================
dfSalnCorr = dfCorr['Salnty'].sort_values(ascending=False)
dfSalnCorr.head(10)   # top 10 positive correlations
dfSalnCorr.tail(10)   # top 10 negative correlations

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
    sns.scatterplot(data=df,x=df[i],y='Salnty', color='Green')
    subPlotNum = subPlotNum + 1

plt.show()


#dfPP = ['R_SALINITY','R_DYNHT','Oxy_µmol/Kg','R_O2']
#dfPP1 = pd.DataFrame(df[['Salnty', 'R_SALINITY', 'NO3uM', 'R_NO3', #'O2ml_L', 'R_O2', 'Oxy_µmol/Kg']])
#sns.pairplot(dfPP1, x_vars=dfPP, y_vars='Salnty')
#===============================================
'''



