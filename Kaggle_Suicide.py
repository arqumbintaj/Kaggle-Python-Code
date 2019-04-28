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

import folium
from folium.plugins import MarkerCluster

pd.set_option('display.max_columns', 80, 'display.max_rows', 50)
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
data=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/suicide-rates.csv')
gps=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/concap.csv')


#  join data frames on Country
data = data.join(gps.set_index('CountryName'), on='country')
data = data.drop(['HDI for year','country-year','CountryCode','CapitalName'], axis=1)

#  check for NULLs
data.isnull().sum()                   #  count of null values
data.dropna(inplace=True)             #  drop rows with null values
data.isnull().sum()                   #  count of null values

#  check for DUPLICATEs
data.duplicated().sum()               #  count of duplicate values
#data.drop_duplicates(inplace = True)  #  drop rows with null values

data.info()
#===============================================


#===============================================
#  2)  Overview of Data
#  male/female sucides over the years (countries -total)
#  male/female sucides over the years (per age group/generation)
#  proportaion of male & female
#  male/female suicide rates for top 10 countries (suicides/100k)
#===============================================
data[['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']].describe()

# countries, continents and years
print('Dataset has', len(data['country'].unique()),'countries on' ,len(data['ContinentName'].unique()),'continents spanning' ,len(data['year'].unique()),'years.')


data[['suicides_no','population','suicides/100k pop','gdp_per_capita ($)']].describe()

# counts
data['age'].unique()
data['age'].value_counts()
data['generation'].value_counts()

# groups
data['suicides/100k pop'].groupby(data['ContinentName']).sum()
data['age'].groupby(data['generation']).value_counts()
data['suicides/100k pop'].groupby(data['sex']).sum()


# suicides rates per country - barplot
suicideRate = data['suicides/100k pop'].groupby(data['country']).mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,20))
plt.title('Suicide Rates per Country (mean=12.5)', fontsize=14)
plt.axvline(x=suicideRate['suicides/100k pop'].mean(),color='gray',ls='--')
sns.barplot(data=suicideRate, y='country',x='suicides/100k pop')

suicideRate.head(5)
"""
Lithuania, Sri Lanka and Hungary top the list with suicide rates much higher than the global mean of 12.5.
"""


#  OVER YEARS - Population, GDP, Suicides and Suicide Rates
YRS = sorted(data.year.unique()-1)  # not including 2016 data
POP = []    # population
GDC = []    # gdp_per_capita ($)
SUI = []    # suicides_no
SUR = []    # suicides/100k pop

for year in sorted(YRS):
    POP.append(data[data['year']==year]['population'].sum())
    GDC.append(data[data['year']==year]['gdp_per_capita ($)'].sum())
    SUI.append(data[data['year']==year]['suicides_no'].sum())
    SUR.append(data[data['year']==year]['suicides/100k pop'].sum())

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
plt.figure(figsize=(10,6))
plt.title('suicides/100k pop vs gdp_per_capita ($)', fontsize=14)
sns.scatterplot(data=data,x='suicides/100k pop',y='gdp_per_capita ($)',hue='age')
"""
GDP per capita has an inverse effect on the rate of suicides; lower the GDP per Capita, higher the rate of suicides.  Rate of people 75+ yrs. are far more likely to commit suicide, and is significantly higher in countries with a lower GDP.
"""

#  CONTINENTS  plot suicides/100k per continent - barplot
fig = plt.figure(figsize=(10,6))
plt.title('Male/Female Suicides/100k per Continents', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data=data, x='sex',y='suicides/100k pop', hue='ContinentName',palette='Blues_r')
"""
Europeans have a higher rate of suicides than any other continents, and males are four times more likely to commit suicides then females.

We cannot assume that this is actually true since this dataset contains only 101 out of 245 countries.
"""

#  SEX vs AGE/GENERATION  plot suicides/100k with generation and sex - barplot
print(data['age'].groupby(data['generation']).value_counts())

fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.title('Male/Female Suicides/100k vs Age', fontsize=14)
plt.xlabel('sex', fontsize=12)
sns.barplot(data=data, x='sex',y='suicides/100k pop', hue='age',palette='hsv')
fig.add_subplot(122)
plt.title('Male/Female Suicides/100k vs Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data=data, x='sex',y='suicides/100k pop', hue='generation')
plt.show()
"""
Males are almost four times more likely to commit suicide then females.  Both males and females over 55 years are more susceptible then other age groups.
"""

#  SUICIDES vs YEAR - lineplot
plt.figure(figsize=(6,6))
plt.title('Suicide Trend - age', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=data,x='year',y='suicides/100k pop',hue='age')
"""
Rate of suicides have been steadily declining since 1995 for all age groups, however, the past few years are seeing an upticks in suicide rates for 55+ age group.
"""

#  MALE/FEMALE SUICIDES vs YEAR - lineplot
fig = plt.figure(figsize=(10,6))
fig.add_subplot(121)
plt.title('Suicide Trend - MALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=data[data['sex'] == 'male'], x='year',y='suicides/100k pop',hue='age')
fig.add_subplot(122)
plt.title('Suicide Trend - FEMALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=data[data['sex'] == 'female'], x='year',y='suicides/100k pop',hue='age')
plt.show()
"""
Rate of suicides are have been declining for both males and females, however, there has been a significant upticks in rates of females in age groups 55+ in recent years.
"""
############################################################


############################################################
#   CORRELATIONS
#       1) setup dataset
#       2) heatmap
############################################################
#  correlation will need "relative" values
#  country, suicides_no, gdp_for_year ($) & year deleted from dataframe
corrData = data.drop(['country','suicides_no',' gdp_for_year ($) ',
                  'population','year'], axis=1)

# rearrange column name so "suicides/100k pop" is first
corrData = data[['suicides/100k pop', 'sex', 'age', 
             'gdp_per_capita ($)','generation', 
             'CapitalLatitude', 'CapitalLongitude',
             'ContinentName']]

#  map sex, age, generation,continent
data['age'].groupby(data['generation']).value_counts()

corrData['sex'] = corrData['sex'].map({'female':0,'male':1})
corrData['age'] = corrData['age'].map({
        '5-14 years':0,'15-24 years':1,'25-34 years':2,
        '35-54 years':3,'55-74 years':4,'75+ years':5})
corrData['generation'] = corrData['generation'].map({
        'Generation Z':0,'Millenials':1,'Generation X':2,
        'Boomers':3,'Silent':4,'G.I. Generation':5})
corrData['ContinentName'] = corrData['ContinentName'].map({
        'Africa':0,'Asia':1,'Australia':2,'Central America':3,
        'Europe':4,'North America':5,'South America':6})

#  Correlations - OVERALL
dataCorr = corrData.corr()
plt.figure(figsize=(8,8))
plt.title('Suicide Correlation', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')

dataCorr['suicides/100k pop'].sort_values(ascending=False)
"""
Overall - as we have seen from the plot and table above, sex and age/generation are the most significant factors when determining the likelihood of someone committing suicide.
"""

#  Correlation MALE - filter dataframe for male/female
dataMale   = corrData[(corrData['sex'] == 1)]                       # male
dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr
corrM = dataMaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation FEMALE - filter dataframe for male/female
dataFemale = corrData[(corrData['sex'] == 0)]                       # female
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


############################################################
#   WORLD MAP
#       1) load dataset
#       2) map with markers
############################################################
"""
Interactive map that displays the suicides_no when the popup is clicked.
"""
#  drop any null lat/lon vales - otherwise folium will fail
#data.info()
#data.isnull().sum()                   #  count of null values
#data.dropna(inplace=True)             #  drop rows with null values

#  create dataframe for mapping
mapData = pd.DataFrame(columns =  ['country','suicides_no','lat','lon'])

mapData.lat = mapData.lat.astype(float).fillna(0.0)
mapData.lon = mapData.lat.astype(float).fillna(0.0)

mapData['country']     = data['suicides_no'].groupby(data['country']).sum().index
mapData['suicides_no'] = data['suicides_no'].groupby(data['country']).sum().values
for i in range(len(mapData.country)):
    mapData.lat[i] =  data.CapitalLatitude[(data['country'] == mapData.country[i])].unique()
    mapData.lon[i] = data.CapitalLongitude[(data['country'] == mapData.country[i])].unique()


#  make map - popup displays country and suicide count
#  lat/lon must be "float"
map = folium.Map(location=[mapData.lat.mean(),mapData.lon.mean()], 
                 tiles='Mapbox Control Room',zoom_start=2)
marker_cluster = MarkerCluster().add_to(map)

"""  popup takes the "join" statement
print(mapData.country[94],"\nsuicides:",format(mapData.suicides_no[94],',d'))
print(''.join([mapData.country[94] + "\nsuicides: " + str(format(mapData.suicides_no[94], ',d'))]))
"""

# TODO - try first print with str( --- )

for i in range(len(mapData)-1):
    popup = ''.join([mapData.country[i] + "\nsuicides: " 
            + str(format(mapData.suicides_no[i], ',d'))])  # commas
    folium.Marker(location=[mapData.lat[i],mapData.lon[i]],
            popup = folium.Popup(popup),
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)

map.add_child(marker_cluster)
map.save("C:\\Users\ACER\Desktop\\mapFolium.html")
map         #  display map
############################################################

