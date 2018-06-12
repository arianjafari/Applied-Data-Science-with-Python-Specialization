
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d400/ee610b9ba00aa5a6ab4f10e417adc6d9254aebd85a7f2e66705199ed.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Ames, Iowa, United States**, and the stations the data comes from are shown on the map below.

# In[118]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib notebook')

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

#leaflet_plot_stations(400,'ee610b9ba00aa5a6ab4f10e417adc6d9254aebd85a7f2e66705199ed')


# In[67]:

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/ee610b9ba00aa5a6ab4f10e417adc6d9254aebd85a7f2e66705199ed.csv')


# In[68]:

df.head()


# In[129]:

df["Year"] = df["Date"].apply(lambda x: x[:4])
df["Month_Date"] = df["Date"].apply(lambda x: x[5:])
#df['Year'], df['Month-Date'] = zip(*df['Date'].apply(lambda x: (x[:4], x[5:]))) efficient way
df.head()


# In[161]:

df_min = df[["Month_Date", "Data_Value"]][df["Element"] == "TMIN"][df["Year"] != "2015"].sort("Month_Date")
df_min = df_min[~df_min["Month_Date"].str.endswith('02-29')].reset_index(drop=True)
df_min = df_min.groupby("Month_Date").aggregate({"Data_Value" : np.min})
df_min.Data_Value = df_min.Data_Value.astype(float)
df_min.head(8)


# In[162]:

df_max = df[["Month_Date", "Data_Value"]][df["Element"] == "TMAX"][df["Year"] != "2015"].sort("Month_Date")
df_max = df_max[~df_max["Month_Date"].str.endswith('02-29')].reset_index(drop=True)
df_max = df_max.groupby("Month_Date").aggregate({"Data_Value" : np.max})
df_max.Data_Value = df_max.Data_Value.astype(float)
#df_max.head()


# In[160]:

df_min_2015 = df[["Month_Date", "Data_Value"]][df["Element"] == "TMIN"][df["Year"] == "2015"].sort("Month_Date")
df_min_2015 = df_min_2015[~df_min_2015["Month_Date"].str.endswith('02-29')].reset_index(drop=True)
df_min_2015 = df_min_2015.groupby("Month_Date").aggregate({"Data_Value" : np.min})
df_min_2015.Data_Value = df_min_2015.Data_Value.astype(float)

df_min_2015.head(8)


# In[151]:

df_max_2015 = df[["Month_Date", "Data_Value"]][df["Element"] == "TMAX"][df["Year"] == "2015"].sort("Month_Date")
df_max_2015 = df_max_2015[~df_max_2015["Month_Date"].str.endswith('02-29')].reset_index(drop=True)
df_max_2015 = df_max_2015.groupby("Month_Date").aggregate({"Data_Value" : np.max})
df_max_2015.Data_Value = df_max_2015.Data_Value.astype(float)
df_max_2015.head()


# In[165]:

brocken_min = np.where(df_min_2015["Data_Value"] < df_min["Data_Value"])[0]
brocken_max =np.where(df_max_2015["Data_Value"] > df_max["Data_Value"])[0]


# In[211]:

from datetime import datetime as dt
temp_fig = plt.figure()
plt.plot(df_max.values, label = "Min Temperature (2004-2015)" , c="r")
plt.plot(df_min.values, label = "Max Temperature (2004-2015)", c="b")
plt.scatter(brocken_max, df_max_2015.iloc[brocken_max], s = 20, c = '#8B0000', label = 'Record Break Maximum (2015)')
plt.scatter(brocken_min, df_min_2015.iloc[brocken_min], s = 20, c = '#00008B', label = 'Record Break Minimum (2015)')
plt.xlabel('Month of the Year')
plt.ylabel('Temperature (Tenths of Degrees C)')
plt.title('Temperature Summary Plot near Ames, IA')
ax = plt.gca()
plt.legend(loc = 4, frameon = False)
plt.gca().fill_between(range(len(df_min)), df_min["Data_Value"], df_max["Data_Value"],facecolor='yellow', alpha=0.25)
plt.ylim([-700, 700])
plt.xticks(range(0, len(df_min),30), (r'Jan', r'Feb', r'Mar', r'Apr', r'May', r'Jun', r'Jul', r'Aug', r'Sep', r'Oct', r'Nov', r'Dec', r'Jan'), rotation='0')

temp_fig.set_size_inches(10, 5)


# In[ ]:



