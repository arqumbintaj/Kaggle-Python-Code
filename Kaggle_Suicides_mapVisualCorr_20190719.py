# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:30:01 2019

@author: Asim Islam


 ---------------------------------------------------------------------------
dataset:    Suicide Rates Overview 1985 to 2016
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
---------------------------------------------------------------------------
Suicides - visualization, correlation and world map
Suicide Rates Overview 1985 to 2016 contains suicide data of 101 countries spanning 32 years.  World capitals GPS contains countries, continents and latitude/longitude information.  I combined the two datasets to look at suicides per continent.

In this analysis, I looked at suicide rates (suicides/100k pop) and GDP per capita (gdp_per_capita ($)) versus countries, continents and over time.  It is divided into following sections:

  - data visualization (plots, plots and more plots)
  - correlations (overall, male and female)
  - world map (trying out Folium package)
  
Attributes:
suicides/100k pop:   combines suicides_no and population
gdp_per_capita ($):  combines GDP and population
years:               2016 data is not complete

Limitations:  Dataset contains 101 countries out of 245, so by far, is not complete but is an excellent resource for learning data visualization :-)


CHORO map
https://geojson-maps.ash.ms/


"""



#===============================================
#  Import Libraries
#===============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
#  Normalize data
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn import model_selection

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
#===============================================


############################################################
#   DATA VISUALIZATION
#       1) load dataset, dataset overview, data cleaning
#       2) overview of data
#       3) correlations
############################################################
#===============================================
#  1)  Load Dataset, Dataset Overview, Data Cleaning
#===============================================
df  = pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/suicide-rates.csv')
gps = pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/concap.csv')


#---------------
#  check for country names in df that are missing in gps
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))

#  update names in df to match the gps file
df.replace({'Cabo Verde':'Cape Verde','Republic of Korea':'South Korea','Russian Federation':'Russia','Saint Vincent and Grenadines':'Saint Vincent and the Grenadines'},inplace=True)
#-------------

#  join data frames on Country
df = df.join(gps.set_index('CountryName'), on='country')
df = df.drop(['HDI for year','country-year','CountryCode','CapitalName'], axis=1)



#===============================================
#  CHECK FOR NULLs
plt.title('DATA - "nulls\"')
sns.heatmap(df.isnull())

nulls = df.isnull().sum().sort_values(ascending = False)
prcet = round(nulls/len(df)*100,2)
df.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])
df.null

df.duplicated().sum()


#-----------------------------
#  Check country names with NULL values
df.country[df['ContinentName'].isnull()].unique()
sorted(df.country.unique())         #  names in data
sorted(gps.CountryName.unique())    #  names in gps

#  check for missing country names
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))

'''
#  update names in df to match the gps file
df.replace({'Cabo Verde':'Cape Verde','Republic of Korea':'South Korea','Russian Federation':'Russia','Saint Vincent and Grenadines':'Saint Vincent and the Grenadines'},inplace=True)

df.country.loc[df['country'] == 'Cabo Verde'] = 'Cape Verde'
df.country.loc[df['country'] == 'Republic of Korea'] = 'South Korea'
df.country.loc[df['country'] == 'Russian Federation'] = 'Russia'
df.country.loc[df['country'] == 'Saint Vincent and Grenadines'] = 'Saint Vincent and the Grenadines'

# check
df[df['country'] == 'Cape Verde'].head()
df[df['country'] == 'China'].head()
df[df['country'] == 'India'].head()
df[df['country'] == 'Pakistan'].head()

#  Top 10 most populous countries in the world
country_list = ['China','India','United States','Indonesia','Brazil','Pakistan','Nigeria','Bangladesh','Russia','Mexico']
df.country[df.country.str.contains('|'.join(country_list))].unique()



print('Dataset has', len(df['country'].unique()),'countries (out of 195) on' ,len(df['ContinentName'].unique()),'continents spanning' ,len(df['year'].unique()),'years.')

#  Top 10 most populous countries in the world
top10 = ['China','India','United States','Indonesia','Brazil','Pakistan','Nigeria','Bangladesh','Russia','Mexico']
in_set = df.country[df.country.str.contains('|'.join(top10))].unique().tolist()

print('Out of the top 10 countries: \n{}\n\nonly the following {} are present \n{}'.format(top10,len(in_set),in_set))

'''
#-----------------------------
#===============================================



#===============================================
#  2)  Overview of Data
#  male/female sucides over the years (countries -total)
#  male/female sucides over the years (per age group/generation)
#  proportaion of male & female
#  male/female suicide rates for top 10 countries (suicides/100k)
#===============================================
df[['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']].describe()

# countries, continents and years
print('Dataset has', len(df['country'].unique()),'countries on' ,len(df['ContinentName'].unique()),'continents spanning' ,len(df['year'].unique()),'years.')


# counts
df['age'].unique()
df['age'].value_counts()
df['generation'].value_counts()

# groups
df['suicides/100k pop'].groupby(df['ContinentName']).sum()
df['age'].groupby(df['generation']).value_counts()
df['suicides/100k pop'].groupby(df['sex']).value_counts()


# suicides rates per country - barplot
suicideRate = df['suicides/100k pop'].groupby(df['country']).mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,20))
plt.title('Suicide Rates per Country (mean={:.2f})'.format(suicideRate['suicides/100k pop'].mean()), fontsize=14)
plt.axvline(x=suicideRate['suicides/100k pop'].mean(),color='gray',ls='--')
sns.barplot(data=suicideRate, y='country',x='suicides/100k pop')

suicideRate.head(5)
"""
Lithuania, Sri Lanka and Russia top the list with suicide rates much higher than the global mean of 12.5.
"""


#  OVER YEARS - Population, GDP, Suicides and Suicide Rates
YRS = sorted(df.year.unique()-1)  # not including 2016 data
POP = []    # population
GDC = []    # gdp_per_capita ($)
SUI = []    # suicides_no
SUR = []    # suicides/100k pop

for year in sorted(YRS):
    POP.append(df[df['year']==year]['population'].sum())
    GDC.append(df[df['year']==year]['gdp_per_capita ($)'].sum())
    SUI.append(df[df['year']==year]['suicides_no'].sum())
    SUR.append(df[df['year']==year]['suicides/100k pop'].sum())

#  plot population and gdp_per_capita ($), 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Population vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.axis('auto')
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,POP)
fig.add_subplot(122)
plt.title('GDP per Capita vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('GDP per Capita (in $)', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,GDC)
plt.show()
"""
Data from 2016 is incomplete and removed from analysis.

Population and GDP per Capita were steadily increasing from 1985 to 2008, and then leveling off and then declining after 2014.  We do not have enough data to see if there is any correlation to the SubPrime market crash of 2008.
"""
#  plot suicides_no and suicides/100k pop, 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Suicides vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUI)
fig.add_subplot(122)
plt.title('Suicides per 100k vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides/100k Population', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUR)
plt.show()
"""
Total number of suicides have been leveling off since the mid-90s, but more importantly, the rate of suicides has been declining since the mid-90s.  It is still difficult to correlate any information between populations, GDP and suicide rates (see correlation section below)
"""


#  SUICIDES/100K vs GDP - scatter
ageList = sorted(df.age.unique())
ageList.remove('5-14 years')
fig = plt.figure(figsize=(12,5))

for i in ageList:
    fig.add_subplot(121)
    plt.title('Suicide Rates per Age Group', fontsize=14)
    plt.xlabel('suicides/100k pop', fontsize=12)
    plt.xlim(0,50)
    plt.legend(ageList)
    df['suicides/100k pop'][df['age'] == i].plot(kind='kde')

    fig.add_subplot(122)
    plt.title('Suicide Rates vs GDP', fontsize=14)
    plt.xlabel('gdp_per_capita ($)', fontsize=12)
    plt.yticks([], [])
    plt.xlim(0,100000)
    #df['gdp_per_capita ($)'][df['age'] == i].plot(kind='kde')
    df['gdp_per_capita ($)'].plot(kind='kde')




"""
GDP per capita has an inverse effect on the rate of suicides; lower the GDP per Capita, higher the rate of suicides.  Rate of people 75+ yrs. are far more likely to commit suicide, and is significantly higher in countries with a lower GDP.
"""

#  CONTINENTS  plot suicides/100k per continent - barplot
fig = plt.figure(figsize=(10,6))
plt.title('Male/Female Suicides/100k per Continents', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data =df, x='sex',y='suicides/100k pop', hue='ContinentName',palette='Blues_r')


"""
Europeans have a higher rate of suicides than any other continents, and males are four times more likely to commit suicides then females.

We cannot assume that this is actually true since this dataset contains only 101 out of 245 countries.
"""

#  SEX vs AGE/GENERATION  plot suicides/100k with generation and sex - barplot
print(df['age'].groupby(df['generation']).value_counts())

fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.title('Male/Female Suicides/100k vs Age', fontsize=14)
plt.xlabel('sex', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='age',palette='hsv')
fig.add_subplot(122)
plt.title('Male/Female Suicides/100k vs Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='generation')
plt.show()
"""
Males are almost four times more likely to commit suicide then females.  Both males and females over 55 years are more susceptible then other age groups.
"""

#  SUICIDES vs YEAR - lineplot
df_sort =  df.sort_values(by='age')  # sort by age
plt.figure(figsize=(10,8))
plt.title('Suicide Trend', fontsize=14)
#plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort,x='year',y='suicides/100k pop',hue='age',ci=None)

"""
Rate of suicides have been steadily declining since 1995 for all age groups, however, the past few years are seeing an upticks in suicide rates for 55+ age group.
"""


#  MALE/FEMALE SUICIDES vs YEAR - lineplot
fig = plt.figure(figsize=(14,6))
fig.add_subplot(121)
plt.title('Suicide Trend - MALE', fontsize=14)
#plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'male'], x='year',y='suicides/100k pop',hue='age',ci=None)
fig.add_subplot(122)
plt.title('Suicide Trend - FEMALE', fontsize=14)
#plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'female'], x='year',y='suicides/100k pop',hue='age',ci=None)
plt.show()

"""
Rate of suicides are have been declining for both males and females, however, there has been a significant upticks in rates of females in age groups 55+ in recent years.
"""


#===========================================
#   Choropleth Maps  - countries in the dataset
#===========================================
#  create dataframe with Country and mean of Suicide rates per 100k Population
df_choro = df[['suicides/100k pop','country']].groupby(['country']).mean().sort_values(by='suicides/100k pop').reset_index()


#  Update US name to match JSON file
df_choro.replace({'United States':'United States of America'},inplace=True)

#  https://www.kaggle.com/ktochylin/world-countries
world_geo = r'C:/Users/ACER/Desktop/JAVA/Kaggle/data/world-countries.json'
world_choropelth = folium.Map(location=[0, 0], tiles='Cartodb Positron',zoom_start=2)

world_choropelth.choropleth(
    geo_data=world_geo,
    data=df_choro,
    columns=['country','suicides/100k pop'],
    key_on='feature.properties.name',
    fill_color='PuBu',  # YlGn
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Suicide Rates per 100k Population')

 
# display map
world_choropelth
world_choropelth.save("C:\\Users\ACER\Desktop\\world_choropelth.html")
#===========================================



#===========================================
#   WORLD MAP
#       1) load dataset
#       2) map with markers
#===========================================
"""
Interactive map that displays the suicides_no when the popup is clicked.
"""
#  create dataframe for mapping
mapdf = pd.DataFrame(columns =  ['country','suicides_no','lat','lon'])

mapdf.lat = mapdf.lat.astype(float).fillna(0.0)
mapdf.lon = mapdf.lat.astype(float).fillna(0.0)

mapdf['country']     = df['suicides_no'].groupby(df['country']).sum().index
mapdf['suicides_no'] = df['suicides_no'].groupby(df['country']).sum().values
for i in range(len(mapdf.country)):
    mapdf.lat[i] =  df.CapitalLatitude[(df['country'] == mapdf.country[i])].unique()
    mapdf.lon[i] = df.CapitalLongitude[(df['country'] == mapdf.country[i])].unique()


#  make map - popup displays country and suicide count
#  lat/lon must be "float"
world_map = folium.Map(location=[mapdf.lat.mean(),mapdf.lon.mean()],zoom_start=2)
marker_cluster = MarkerCluster().add_to(world_map)

"""  popup takes the "join" statement
print(mapdf.country[94],"\nsuicides:",format(mapdf.suicides_no[94],',d'))
print(''.join([mapdf.country[94] + "\nsuicides: " + str(format(mapdf.suicides_no[94], ',d'))]))
"""

'''
for i in range(len(mapdf)-1):
    popup = ''.join([mapdf.country[i] + "\nsuicides: " 
            + str(format(mapdf.suicides_no[i], ',d'))])  # commas
    folium.Marker(location=[mapdf.lat[i],mapdf.lon[i]],
            popup = folium.Popup(popup),
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)
'''


for i in range(len(mapdf)-1):
    label = '{}:  {} suicides'.format(mapdf.country[i].upper(),mapdf.suicides_no[i])
    label = folium.Popup(label, parse_html=True)
    folium.Marker(location=[mapdf.lat[i],mapdf.lon[i]],
            popup = label,
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)


world_map.add_child(marker_cluster)
world_map.save("C:\\Users\ACER\Desktop\\world_mapFolium.html")
world_map         #  display map
#===========================================
############################################################


'''
#  NEED NORMALIZATION


#  remove commas and save as float64
df_norm[' gdp_for_year ($) '] = df_norm[' gdp_for_year ($) '].str.replace(',','').astype('float64')

df_norm[['suicides_no', 'population','suicides/100k pop', ' gdp_for_year ($) ', 'gdp_per_capita ($)']] = MinMaxScaler().fit_transform(df_norm[['suicides_no', 'population','suicides/100k pop', ' gdp_for_year ($) ', 'gdp_per_capita ($)']])

for i in df_norm.columns:
    df[i] = df_norm[i]
'''

############################################################
#  ENCODING and NORMALIZATION
df_orig = df.copy(deep=True)
#   df = df_orig.copy(deep=True)
############################################################
df_corr = df.drop(['country','year','CapitalLatitude','CapitalLongitude'], axis=1)

#  rearrange column name so "suicides/100k pop" is first
# remove ""suicides_no"
df_corr = df[['suicides/100k pop', 'sex', 'age', 'population',' gdp_for_year ($) ','gdp_per_capita ($)', 'generation','suicides_no','ContinentName']]


#  one-hot encoding - manual
#===========================================
df_corr['sex'] = df_corr['sex'].map({'female':0,'male':1})
df_corr['age'] = df_corr['age'].map({
        '5-14 years':0,'15-24 years':1,'25-34 years':2,
        '35-54 years':3,'55-74 years':4,'75+ years':5})
df_corr['generation'] = df_corr['generation'].map({
        'Generation Z':0,'Millenials':1,'Generation X':2,
        'Boomers':3,'Silent':4,'G.I. Generation':5})
df_corr['ContinentName'] = df_corr['ContinentName'].map({
        'Africa':0,'Asia':1,'Australia':2,'Central America':3,
        'Europe':4,'North America':5,'South America':6})

#  remove commas and save as float64
df_corr[' gdp_for_year ($) '] = df_corr[' gdp_for_year ($) '].str.replace(',','').astype('float64')
#===========================================
#df_corr.describe(include=['O'])   #  CATEGORICAL DATA
df_corr.info()

#  normalize
#===========================================
df_norm = MinMaxScaler().fit_transform(df_corr)

#  create dataframe
df = pd.DataFrame(df_norm, index=df_corr.index, columns=df_corr.columns)
############################################################



############################################################
#   CORRELATIONS
#       1) setup dataset
#       2) heatmap
############################################################
#  Correlations - OVERALL
dataCorr = df.corr()
plt.figure(figsize=(8,8))
plt.title('Suicide Correlation', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')

dataCorr['suicides/100k pop'].sort_values(ascending=False)
"""
Overall - as we have seen from the plot and table above, sex and age/generation are the most significant factors when determining the likelihood of someone committing suicide.
"""

#  Correlation MALE - filter dataframe for male/female
dataMale   = df[(df['sex'] == 1)]                       # male
dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr
corrM = dataMaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation FEMALE - filter dataframe for male/female
dataFemale = df[(df['sex'] == 0)]                       # female
dataFemaleCorr = dataFemale.drop(["sex"], axis=1).corr()    # female corr
corrF = dataFemaleCorr['suicides/100k pop'].sort_values(ascending=False)

# print for Kaggle 
print("Correlation - FEMALE:\n", dataFemaleCorr['suicides/100k pop'].sort_values(ascending=False))
print("Correlation - FEMALE:\n", dataFemaleCorr['suicides/100k pop'].sort_values(ascending=False))

#  Correlation heatmaps for FEMALE/MALE
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
plt.title('Suicide Correlation - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
fig.add_subplot(122)
plt.title('Suicide Correlation - FEMALE ', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
plt.show()

#  Correlation - sorted for both male/female
corrALL = pd.DataFrame(columns = ['MALE','correlation-m','FEMALE','correlation-f'])
corrALL['MALE']   = corrM.index
corrALL['correlation-m'] = corrM.values
corrALL['FEMALE'] = corrF.index
corrALL['correlation-f'] = corrF.values
print(corrALL)
"""
Age/generation is the primary factor in determining the likelihood of a male or a female committing suicide.
"""
############################################################



################################################
#  MACHINE LEARNING
#   - Split Dataset Into Train and Test
#   - Define the Models
#   - Evaluate the Models
#  -  target:   suicides/100k pop
################################################
#-----------------------------------------------
#  Split Dataset Into Train and Test
#-----------------------------------------------
seed = 7
X = df.drop(['suicides/100k pop'], axis = 1)
y = df['suicides/100k pop']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=seed)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
#-----------------------------------------------

from sklearn import linear_model

LR = linear_model.LinearRegression()
train_y_ = LR.fit(X_train, y_train)
# The coefficients
print ('Coefficients: ', LR.coef_)
print ('Intercept: ',LR.intercept_)

import numpy as np
from sklearn.metrics import r2_score
y_test_PRED = LR.predict(X_test)
X_test
print("\ndegree=2:\tMean absolute error: %.2f" % np.mean(np.absolute(y_test_PRED - y_test)))
print("degree=2:\tResidual sum of squares (MSE): %.2f" % np.mean((y_test_PRED - y_test) ** 2))
print("degree=2:\tR2-score: %.2f" % r2_score(y_test_PRED , y_test))





