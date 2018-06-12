
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:

import pandas as pd
import re

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)


# In[2]:

df.tail(10)


# In[145]:

from datetime import datetime
month_dic = {"Jan" : "01", "Feb" : "02", "Apr" : "03", "Mar" : "04", "May" : "05", "Jun" : "06", "Jul" : "07", "Aug" : "08", "Sep" : "09", "Oct" : "10", "Nov" : "11", "Dec" : "12"}
def date_sorter():
    res = []
    index_set = set([])
    for idx,  row in df.iteritems():
        
        date = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", row)
        
        if len(date) != 0:
            if idx not in index_set:
                if date[0][-3].isdigit():
                    date[0]=date[0].replace("/", "-")
                    res.append((idx,date[0]))
                    index_set.add(idx)
                else:
                    date[0]=date[0].replace("/", "-")
                    res.append((idx,date[0][:-2] + "19" + date[0][-2:]))
                    index_set.add(idx)
            
        #date = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2}", row)
        #if len(date) != 0:
            #res.append((idx,date[0][:-2] + "19" + date[0][-2:]))
            
        date = re.findall(r"(?:\d{1,2}[ -])?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,]*[ ,-](?:\d{1,2}[thstnd,]*[ -])?\d{2,4}", row)    
        if len(date) != 0:
            date[0]=date[0].replace(",", "")
            list1 = re.split(" ", date[0][:-4].strip())
            
            list1.sort()
            
            if len(list1) == 2:
                
                res.append((idx,month_dic[list1[1][0:3]] + "-" + list1[0] + "-" + date[0][-4:]))
                index_set.add(idx)
            else:
                res.append((idx, month_dic[list1[0][0:3]] + "-01-" + date[0][-4:]))
                index_set.add(idx)
        
        date = re.findall(r"\d{1,2}[/-]\d{4}", row)    
        if len(date) != 0:
            if idx not in index_set:
                list2 = re.split("/", date[0].strip())
                res.append((idx,list2[0] + "-01-" +list2[1]))
                index_set.add(idx)
        date = re.findall(r"\d{4}", row)    
        if len(date) != 0:
            if idx not in index_set:
                res.append((idx,"01-01-" + date[0][-4:]))
                index_set.add(idx)
        #print(res)
        #break
        res_new = []
        for item in res:
            #print(item[0])
            res_new.append((item[0], datetime.strptime(item[1], '%m-%d-%Y')))
         
        res_new.sort(key = lambda x : x[1])
        
        result = []
        for item in res_new:
            result.append(item[0])
    
    return pd.Series(result)
#date_sorter()

