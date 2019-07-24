# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:37:48 2019

@author: ACER
"""

'''
ArcGIS
Folium World Maps
Choropleth Maps


suicide database
world-countries.json
'''

#===========================================
#  load libraries
#===========================================
import pandas as pd

import geocoder
import folium
from folium.plugins import MarkerCluster

# %matplotlib inline    #  for jupyter
import warnings
warnings.filterwarnings('ignore')

#  Kaggle directories
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
#===========================================



#===========================================
#   ARCGIS
#===========================================
geocoder.arcgis('dallas, texas').latlng
'''
#  Define function to get latitude & longitude
def arc_latlng(location):
    lat_lng_coords = None
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}'.format(location))
        lat_lng_coords = g.latlng
    return lat_lng_coords
'''

def arc_latlng(location):
    g = geocoder.arcgis('{}'.format(location))
    lat_lng_coords = g.latlng
    print(location,lat_lng_coords)
    return lat_lng_coords

arc_latlng('dallas, texas')     #  test arc_latlng


arc_latlng('dallas, texas')     #  test arc_latlng
arc_latlng('everest')     #  test arc_latlng

#  ARCGIS can take location names, zip/postal codes, landmark, etc.
#  location list
#  10001 is the zip code of Manhattan, New York, US
#  M9B   is a postal code in Toronto, Canada
#  Everest is Mount. Everest in Nepal
location = ['10001','Tokyo','Sydney','Beijing','Karachi','Dehli', 'Everest','M9B','Eiffel Tower','Sao Paulo','Moscow']


#0902iuey7

#  call arc_latlng function
loc_latlng = [arc_latlng(location) for location in location]
cleanedList = [x for x in loc_latlng if x != 'nan']


#  create dataframe for the results
df = pd.DataFrame(data = loc_latlng, columns = {'Latitude','Longitude'})
#  sometimes dataframe flips the order of columns
df.columns = ['Latitude','Longitude']
#  add location to dataframe
df['Location'] = location

df

# 3 & 4 are invalid
invalid_loc = ['london','berlin','0902iuey7','999paris']
invalid_latlng = [arc_latlng(invalid_loc) for invalid_loc in invalid_loc]
#===========================================



#===========================================
#  2. Folium Maps
#===========================================

#  2.1  Folium - World Map
#  center map on mean of Latitude/Longitude
map_world = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],zoom_start = 2)

#  add Locations to map
for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world)

#  display interactive map
map_world
map_world.save("C:\\Users\ACER\Desktop\\map_world.html")




#  2.2  Folium Markers
locationNYC = ['Empire State Building','Central Park','Wall Street','Brooklyn Bridge','Statue of Liberty','Rockefeller Center', 'Guggenheim Museum','Metlife Building','Times Square','United Nations Headquarters','Carnegie Hall']
locNYC_latlng = [arc_latlng(locationNYC) for locationNYC in locationNYC]

dfNY = pd.DataFrame(data = locNYC_latlng, columns = {'Latitude','Longitude'})
dfNY.columns = ['Latitude','Longitude']
dfNY['Location'] = locationNYC
dfNY



map_world_NYC = folium.Map(location=[df.Latitude.mean(), df.Longitude.mean()],tiles = 'openstreetmap', zoom_start = 2)

for lat, lng, label in zip(df.Latitude, df.Longitude, df.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world_NYC)

marker_cluster = MarkerCluster().add_to(map_world_NYC)
for lat, lng, label in zip(dfNY.Latitude, dfNY.Longitude, dfNY.Location):
    folium.Marker(location=[lat,lng],
            popup = label,
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)

map_world_NYC.add_child(marker_cluster)
map_world_NYC.save("C:\\Users\ACER\Desktop\\map_world_NYC.html")
map_world_NYC         #  display map
#===========================================



#===========================================
#  3. Choropleth Maps
#===========================================
dfs = pd.read_csv('C:/Users/ACER/Desktop/JAVA/Kaggle/data/suicide-rates.csv')

#  Need Country. Year and suicides/100k population columns
dfs[dfs['year'] == 2013]
dfs = dfs[dfs['year'] == 2013]
dfs = dfs[['country','year','suicides/100k pop']].groupby('country').sum()
dfs.reset_index(inplace=True)
dfs.head()

dfs.replace({
        'United States':'United States of America',
        'Republic of Korea':'South Korea',
        'Russian Federation':'Russia'},
        inplace=True)


world_geo = r'C:/Users/ACER/Desktop/JAVA/Kaggle/data/world-countries.json'

world_choropelth = folium.Map(location=[0, 0], tiles='Cartodb Positron',zoom_start=2)

world_choropelth.choropleth(
    geo_data=world_geo,
    data=dfs,
    columns=['country','suicides/100k pop'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Suicide rates per 100k Population (2015)')

# display map
world_choropelth
world_choropelth.save("C:\\Users\ACER\Desktop\\world_choropelth.html")

'''
fill_color: string, default 'blue'
'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'RdPu','YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

tiles
'cartodbdark_matter',
'cartodbpositron',
'cartodbpositronnolabels',
'cartodbpositrononlylabels',
'openstreetmap',
'stamenterrain',
'stamentoner',
'stamentonerbackground',
'stamentonerlabels',
'stamenwatercolor',
'''
#===========================================


Location = ['10001','Tokyo','Sydney','Beijing','Karachi', 'Dehli','Everest','M9B','Eiffel Tower','Sao Paulo','Moscow']
Latitude = [40.74876000000006, 35.68945633200008, -33.869599999999934, 39.90750000000003, 24.90560000000005, 28.653810000000078, 27.987910000000056, 43.64969222700006, 48.85859991892235, -23.562869999999975, 55.75696000000005]
Longitude = [-73.99331999999998, 139.69171608500005, 151.2069100000001, 116.39723000000004, 67.08220000000006, 77.22897000000006, 86.92529000000007, -79.55394499999994, 2.293980070546176, -46.654679999999985, 37.61502000000007]

df = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
#  sometimes dataframe flips the order of columns
df.columns = ['Location','Latitude','Longitude']
#  add location to dataframe
df['Latitude'] = Latitude
df['Longitude'] = Longitude

LocationNY = ['Empire State Building', 'Central Park', 'Wall Street', 'Brooklyn Bridge', 'Statue of Liberty', 'Rockefeller Center', 'Guggenheim Museum', 'Metlife Building', 'Times Square', 'United Nations Headquarters', 'Carnegie Hall']

LatitudeNY = [40.74837000000008, 40.76746000000003, 40.705790000000036, 40.70765000000006, 40.68969000000004, 40.758290000000045, 40.78300000000007, 40.75407000000007, 40.75648000000007, 40.74967000000004, 40.76494993060773]

LongitudeNY = [-73.98463999999996, -73.97070999999994, -74.00987999999995, -73.99890999999997, -74.04358999999994, -73.97750999999994, -73.95899999999995, -73.97637999999995, -73.98617999999993, -73.96916999999996, -73.9804299522477]

dfNY = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
#  sometimes dataframe flips the order of columns
dfNY.columns = ['Location','Latitude','Longitude']
#  add location to dataframe
dfNY['Location'] = LocationNY
dfNY['Latitude'] = LatitudeNY
dfNY['Longitude'] = LongitudeNY
















