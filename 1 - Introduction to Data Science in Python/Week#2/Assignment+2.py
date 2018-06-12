
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 2 - Pandas Introduction
# All questions are weighted the same in this assignment.
# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on [All Time Olympic Games Medals](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table), and does some basic data cleaning. 
# 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.

# In[2]:

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index)

df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()


# ### Question 0 (Example)
# 
# What is the first country in df?
# 
# *This function should return a Series.*

# In[3]:

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 


# ### Question 1
# Which country has won the most gold medals in summer games?
# 
# *This function should return a single string value.*

# In[4]:

def answer_one():
    most_gold = df.where(df['Gold'] == max(df['Gold']))
    most_gold = most_gold.dropna()
    return most_gold.index[0]


# ### Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# 
# *This function should return a single string value.*

# In[5]:

def answer_two():
    biggest_diff = df.where(abs(df['Gold'] - df['Gold.1']) == max(abs(df['Gold'] - df['Gold.1'])))
    biggest_diff =biggest_diff.dropna()
    return biggest_diff.index[0]


# ### Question 3
# Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count? 
# 
# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
# 
# Only include countries that have won at least 1 gold in both summer and winter.
# 
# *This function should return a single string value.*

# In[6]:

def answer_three():
    ratio_column = abs(df['Gold'][df["Gold"] > 0] - df['Gold.1'][df["Gold.1"] > 0]) /(df['Gold'][df["Gold"] > 0] + df['Gold.1'][df["Gold.1"] > 0])
    max_diff_val = max(ratio_column.dropna())
    biggest_ratio = df.where(abs(df['Gold'][df["Gold"] > 0] - df['Gold.1'][df["Gold.1"] > 0]) /(df['Gold'][df["Gold"] > 0] + df['Gold.1'][df["Gold.1"] > 0]) == max_diff_val)
    biggest_ratio =biggest_ratio.dropna()
    return biggest_ratio.index[0]


# ### Question 4
# Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.
# 
# *This function should return a Series named `Points` of length 146*

# In[7]:

def answer_four():
    
    return pd.Series(df["Gold.2"] * 3 + df["Silver.2"] * 2 + df["Bronze.2"], index = df.index)


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau](http://www.census.gov). Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. [See this document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2015/co-est2015-alldata.pdf) for a description of the variable names.
# 
# The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.
# 
# ### Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)
# 
# *This function should return a single string value.*

# In[8]:

census_df = pd.read_csv('census.csv')
census_df.head()


# In[9]:

state_county = []
def answer_five():
    states = list(set(census_df["STNAME"]))
    for state in states:
        number_of_county = len(census_df["COUNTY"][census_df["STNAME"]== state]) - 1
        state_county.append((number_of_county,state))
        
        
    max_county_state = max(state_county)[1]    
    return max_county_state
answer_five()


# ### Question 6
# **Only looking at the three most populous counties for each state**, what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.
# 
# *This function should return a list of string values.*

# In[10]:

top_pop_state = []
def answer_six():
    states = list(set(census_df["STNAME"]))
    for state in states:
        county_pop = census_df["CENSUS2010POP"][census_df["STNAME"]== state]
        three_top_pop_county = sorted(county_pop, reverse = True)[1:4]
        sum_top_three = sum(three_top_pop_county)
        top_pop_state.append((sum_top_three, state))
    
    top_three_pop_state = sorted(top_pop_state, reverse=True)[0:3]
    top_state_name = []
    for item in top_three_pop_state:
        top_state_name.append(item[1])
        
    return top_state_name


# ### Question 7
# Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
# 
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.
# 
# *This function should return a single string value.*

# In[11]:

def answer_seven():

    max_county_diff_list = []

    year_list = []
    
    for idx in range(6):
        year_list.append("POPESTIMATE201" + str(idx))
    
    for index, row in census_df.iterrows():
        
        if row.STNAME != row.CTYNAME:
            county_pop = []
            for year in year_list:
                county_pop.append(census_df[year][index])
            
            county_pop = sorted(county_pop)
            max_county_diff_list.append(((county_pop[-1] - county_pop[0]),row.CTYNAME))
    
    max_county_diff = max(max_county_diff_list)[1]

    return max_county_diff
answer_seven()


# ### Question 8
# In this datafile, the United States is broken up into four regions using the "REGION" column. 
# 
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
# 
# *This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).*

# In[14]:

def answer_eight():
    
    region_1_2_df = census_df[(census_df['REGION'] == 1) | (census_df['REGION'] == 2)]
    frame = []
    frame_idx = []
    for idx, row in region_1_2_df.iterrows():
        if row.CTYNAME.find("Washington") == 0:
            if row.POPESTIMATE2015 > row.POPESTIMATE2014:
                frame.append([row.STNAME, row.CTYNAME])
                frame_idx.append(idx)
    result = pd.DataFrame(frame, columns=['STNAME', 'CTYNAME'], index=frame_idx)
    return result
answer_eight()


# In[ ]:



