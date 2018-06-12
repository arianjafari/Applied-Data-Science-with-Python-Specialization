
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[2]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[27]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[68]:

zillow = pd.read_csv("City_Zhvi_AllHomes.csv")

GDP = pd.read_excel("gdplev.xls", skiprows= 7, parse_cols = "A:C, E:G")
GDP.columns = ["year", "year GDP", "2009 GDP", "year Qrt", "Qrt GDP", "2009 Qrt GDP"]
my_gdp = GDP[GDP['year Qrt'] >= "2000"].reset_index(drop=True).drop(['year', "year GDP", "2009 GDP","Qrt GDP"], 1)
GDP.head()


# In[61]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    
    state_set = {'Ohio' ,'Kentucky', 'American Samoa', 'Nevada', 'Wyoming', 'National', 'Alabama', 'Maryland', 'Alaska', 'Utah',  'Oregon', 'Montana',  'Illinois',  'Tennessee', 'District of Columbia',  'Vermont','Idaho',  'Arkansas',  'Maine',  'Washington', 'Hawaii', 'Wisconsin', 'Michigan', 'Indiana', 'New Jersey', 'Arizona', 'Guam',  'Mississippi', 'Puerto Rico', 'North Carolina', 'Texas',  'South Dakota', 'Northern Mariana Islands', 'Iowa',  'Missouri',  'Connecticut',  'West Virginia', 'South Carolina', 'Louisiana',  'Kansas', 'New York',  'Nebraska',  'Oklahoma',  'Florida',  'California', 'Colorado',  'Pennsylvania', 'Delaware',  'New Mexico',  'Rhode Island',  'Minnesota',  'Virgin Islands',  'New Hampshire',  'Massachusetts',  'Georgia', 'North Dakota',  'Virginia'}
    uni_town_list = []
    with open('university_towns.txt') as uni_town:
        for line in uni_town:
            row = line.split("[")[0].split("(")[0]
    
            if row in state_set:
                state_name = row
            else:
                uni_town_list.append([state_name.strip() , row.strip()])
            
            
    unitowns = pd.DataFrame(uni_town_list, columns=["State", "RegionName"])
    return unitowns


# In[65]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    
    
    for index, row in GDP.iterrows():
        if (index < len(GDP) - 2)  and float(row["year Qrt"][:-2]) > 2000:
            if (GDP["Qrt GDP"].iloc[index+2] < GDP["Qrt GDP"].iloc[index+1]) and (GDP["Qrt GDP"].iloc[index+1] < GDP["Qrt GDP"].iloc[index]):
                return GDP["year Qrt"].iloc[index]
        
    return ""
get_recession_start()


# In[66]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''

    
    recession_start = get_recession_start()
    
    for index, row in GDP.iterrows():
        if (index > 1 and index < len(GDP) - 2)  and float(row["year Qrt"][:-2]) > 2000:
            if (GDP["Qrt GDP"].iloc[index+2] > GDP["2009 Qrt GDP"].iloc[index+1]) and (GDP["Qrt GDP"].iloc[index+1] > GDP["Qrt GDP"].iloc[index]) and (GDP["Qrt GDP"].iloc[index - 1] >= GDP["Qrt GDP"].iloc[index]):        
                if GDP["year Qrt"].iloc[index+2] > recession_start:
                    return GDP["year Qrt"].iloc[index+2]
       
    return "ANSWER"
get_recession_end()


# In[67]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    
    recession_start = get_recession_start()
    recession_end = get_recession_end()
    
    bottom = GDP["Qrt GDP"][GDP["year Qrt"] < recession_end][GDP["year Qrt"] > recession_start].min()
    
    
    
    bottom_year = GDP.loc[GDP["Qrt GDP"] == bottom, "year Qrt"].item()
    
    return bottom_year
get_recession_bottom()


# In[8]:

zillow = pd.read_csv("City_Zhvi_AllHomes.csv")


# In[9]:

print(len(zillow))
zillow.head()


# In[31]:

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    columns = list(zillow.columns)
    columns_19s = []
    columns_no_need = ["RegionID", "Metro", "CountyName", "SizeRank"]
    
    for idx in range(5, len(columns)):    # finding the 19s columns
        if columns[idx][:4] < "2000":
            columns_19s.append(columns[idx])
           
    for col in columns_19s:            # removing 19s values from the columns list 
        columns.remove(col)
    
    for col in columns_no_need:       # removing "RegionID", "Metro", "CountyName", "SizeRank" from the columns list
        columns.remove(col)
        
    new_zillow = zillow[columns].copy()
    quarter_zillow = new_zillow[["RegionName", "State"]].copy()
    quarter_zillow["State"].replace(states, inplace=True)
    
    #quarter_zillow = quarter_zillow.replace({'State':states})  # this is an alternative way
    
    
    for val in range(2000,2016):
        
        quarter_zillow[str(val) + "q1"] = new_zillow[[str(val)+"-01", str(val)+"-02", str(val)+"-03"]].mean(axis=1)
        quarter_zillow[str(val) + "q2"] = new_zillow[[str(val)+"-04", str(val)+"-05", str(val)+"-06"]].mean(axis=1)
        quarter_zillow[str(val) + "q3"] = new_zillow[[str(val)+"-07", str(val)+"-08", str(val)+"-09"]].mean(axis=1)
        quarter_zillow[str(val) + "q4"] = new_zillow[[str(val)+"-10", str(val)+"-11", str(val)+"-12"]].mean(axis=1)
    
    year = 2016
    
    quarter_zillow[str(year) + "q1"] = new_zillow[[str(year)+"-01", str(year)+"-02", str(year)+"-03"]].mean(axis=1)
    quarter_zillow[str(year) + "q2"] = new_zillow[[str(year)+"-04", str(year)+"-05", str(year)+"-06"]].mean(axis=1)
    quarter_zillow[str(year) + "q3"] = new_zillow[[str(year)+"-07", str(year)+"-08"]].mean(axis=1)
    
    quarter_zillow = quarter_zillow.set_index(['State','RegionName'])
    
    return quarter_zillow
convert_housing_data_to_quarters().head()


# In[155]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    unitown = get_list_of_university_towns()
    quarter_zillow = convert_housing_data_to_quarters()
    recession_start = get_recession_start()
    recession_end = get_recession_end()
    bottom_year = get_recession_bottom()
    
    rec_bot_columns = []
    #columns = list(quarter_zillow.columns) 
    
    for col in list(quarter_zillow.columns):
        if col >= recession_start and col <= bottom_year:
            rec_bot_columns.append(col)
    
    rec_bot_data = quarter_zillow[rec_bot_columns].copy()
    
    unitown_list = []
    for idx, row in unitown.iterrows():
        unitown_list.append((row["State"], row["RegionName"]))
    
    nonunitown_list = []
    
    for idx, row in rec_bot_data.iterrows():

        if idx not in unitown_list:
            nonunitown_list.append(idx)

            
    
    unitown_price = rec_bot_data.loc[unitown_list]   
    unitown_price.index.set_names(["State", "RegionName"], inplace=True)
    unitown_price["diff"] = unitown_price[bottom_year] - unitown_price[recession_start]
        
    
    non_unitown_price = rec_bot_data.loc[nonunitown_list]
    non_unitown_price["diff"] = non_unitown_price[bottom_year] - non_unitown_price[recession_start]
    
    t,p = ttest_ind(unitown_price['diff'].dropna(),non_unitown_price['diff'].dropna())
    
    different = True if p < 0.01 else False

    better = "non-university town" if unitown_price['diff'].mean() < non_unitown_price['diff'].mean() else "university town"
    
    return different, p, better
run_ttest()


# In[ ]:



