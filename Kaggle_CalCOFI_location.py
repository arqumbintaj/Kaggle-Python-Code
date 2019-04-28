# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:40:54 2019

@author: Asim Islam

How to Plot Lat/Longitude on an Interactive World Map
Plotting Latitude and Longitude on an Interactive World Map

This is a short tutorial/quick start guide on how to plot location-based data points (in Latitude and Longitude "decimals") on an interactive world map using the folium package in Python.

NOTE:  folium takes DECIMAL values for Latitude and Longitude.  Use the formula if Latitude and Longitude are only available in degree/minutes/seconds:
    
    Latitude  Decimal = degrees + (minutes/60) + (seconds/3600)
    Longitude Decimal = degrees + (minutes/60) + (seconds/3600)


Jupyter supports folium.  For other applications (e.g. PyCharm, Spyder,etc.), use "pip install folium --upgrade" or respective "install" variations.

I will keep adding more info/code as I go along.  Upvote if you found this useful :-)

------------------------------------------------------------------
data SET:   CalcofiLOC  (https://www.kaggle.com/sohier/calCOFI) ------------------------------------------------------------------
"""

#===============================================
#  Import Libraries
#===============================================
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#  Load the Dataset
cofiLOC=pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/Calcofi/cast.csv')

# only need Lat_Dec, Lon_Dec and Date
cofiLOC = cofiLOC[['Lat_Dec', 'Lon_Dec','Date']]

#  cast.csv has 34,403 location points.
#  Using a subset of location points for this tutorial
#    NOTE: reset index, otherwise "for loop" will not work with
#          cofiLOC.Lat_Dec[index] and cofiLOC.Lon_Dec[index]
cofiLOC = cofiLOC.tail(1000)
cofiLOC = cofiLOC.reset_index(drop=True)  # reset index 

#  NOTE:  reset_index is required, otherwise the "[i]" in the "for loop" will error.
#===============================================


#===============================================
#   Folium MarkerCluster
#===============================================
# folium for location mapping
import folium
from folium.plugins import MarkerCluster

#  Make an empty world map, 
#  zoom into the mean Latitude and Longitude (type: float)
#  zoom_start=0 is the world map
map1 = folium.Map(location=[cofiLOC.Lat_Dec.mean(),cofiLOC.Lon_Dec.mean()], zoom_start=7)

marker_cluster = MarkerCluster().add_to(map1)

#  Add all datapoints to map.  Add "Dates" to popups
for i in range(len(cofiLOC)):
    folium.Marker(location=[cofiLOC.Lat_Dec[i],cofiLOC.Lon_Dec[i]],
            popup = folium.Popup("boo {}",cofiLOC.Date[i]),
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)

map1.add_child(marker_cluster)

#map.save("C:\<local machine>\mapFolium.html")  # save to desktop
map1.save("C:\\Users\ACER\Desktop\\map1.html")

#  Render interactive map in Jupyter (time-intensive)
map1
#===============================================

"""
cities = pd.DataFrame(columns={'CITY','POPULATION'})
cities.CITY = ['Tokyo', 'Delhi','Shanghai','Sao Paulo',
               'Mexico City','Cairo','Dhaka','Mumbai',
               'Beijing','Osaka','Karachi','Chongqing',
               'Buenos Aires','Istanbul','Kolkata',
               'Lagos','Manila','Tianjin','Rio De Janeiro',
               'Guangzhou','Moscow','Lahore','Shenzhen',
               'Bangalore','Paris',]

cities.POPULATION = ['37435191','29399141','26317104','21846507',
                     '21671908','20484965','20283552','20185064',
                     '20035455','19222665','15741406','15354067',
                     '15057273','14967667','14755186','13903620',
                     '13698889','13396402','13374275','12967862',
                     '12476171','12188196','12128721','11882666',
                     '10958187']


#  change type to int and use comma separators
cities['POPULATION'] = cities['POPULATION'].apply(np.int)
cities['POPULATION'] = cities['POPULATION'].apply('{:,}'.format)
cities.info()

map2 = folium.Map(zoom_start=0)


marker_cluster = MarkerCluster().add_to(map2)

#  Add all datapoints to map.  Add "Dates" to popups
for i in range(len(cofiLOC)):
    folium.Marker(location=[cofiLOC.Lat_Dec[i],cofiLOC.Lon_Dec[i]],
            popup = (cofiLOC.Date[i]),
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)

map2.add_child(marker_cluster)

#map.save("C:\<local machine>\mapFolium.html")  # save to desktop
map2.save("C:\\Users\ACER\Desktop\\map1.html")

#  Render interactive map in Jupyter (time-intensive)
map2
"""