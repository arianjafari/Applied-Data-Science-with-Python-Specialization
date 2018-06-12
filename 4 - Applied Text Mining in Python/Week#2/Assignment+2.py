
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[26]:

import nltk
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# In[24]:

text1[:10]


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[27]:

def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[28]:

def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[29]:

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[30]:

def answer_one():
    
    lex_div = len(set(text1)) / len(text1)
    
    return lex_div

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[38]:

def answer_two():
    
    count = 0
    for word in text1:
        if (word == "whale" or word == "Whale"):
            
            count += 1
    
    return float(count/len(text1)*100)

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[33]:

def answer_three():
    
    import operator
    from nltk import FreqDist
    
    dist = FreqDist(text1)
    
    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_dist[:20]

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return a sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[36]:

def answer_four():
    
    from nltk import FreqDist
    
    dist = FreqDist(text1)
    
    my_list = [w for w in list(dist.keys()) if (len(w) > 5 and dist[w] > 150)]
    return sorted(my_list)

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[40]:

def answer_five():
    
    longest_word = max(text1, key= lambda x: len(x))
    return (longest_word, len(longest_word))

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[45]:

def answer_six():
    
    from nltk import FreqDist
    
    dist = FreqDist(text1)
    
    my_list = [(dist[w], w) for w in list(dist.keys()) if (w.isalpha() and dist[w] > 2000)]
    
   
    return sorted(my_list, key = lambda x : x[0] , reverse = True)

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[51]:

def answer_seven():
    
    
    return len(text1) / len(nltk.sent_tokenize(moby_raw))

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[55]:

def answer_eight():
    
    from collections import Counter
    import operator
    
    list1 = nltk.pos_tag(text1)
    freq  = Counter(elem[1] for elem in list1)
    
       
    sorted_dist = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_dist[:5]

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[58]:

from nltk.corpus import words

correct_spellings = words.words()
print(correct_spellings[-10:])


# In[68]:

from nltk.metrics.distance import jaccard_distance , edit_distance
word1 = "cormulent"
word2 = "a"
print(set(word1))
print(set(word2))
jaccard_distance(set(word1),set([]))
rr = nltk.ngrams(word1, n=3)
rr


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[75]:

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import jaccard_distance
    result = []
    for entry in entries:
        
        word_list = [word for word in correct_spellings if(word[0] == entry[0])]
        
        dist = [(word, nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)),set(nltk.ngrams(word, n=3)))) for word in word_list]
           
        result.append(sorted(dist, key = lambda x: x[1])[0][0])
        
    return result
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[76]:

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import jaccard_distance
    result = []
    for entry in entries:
        
        word_list = [word for word in correct_spellings if(word[0] == entry[0])]
        
        dist = [(word, nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)),set(nltk.ngrams(word, n=4)))) for word in word_list]
           
        result.append(sorted(dist, key = lambda x: x[1])[0][0])
        
    return result
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[77]:

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import edit_distance
    result = []
    for entry in entries:
        
        word_list = [word for word in correct_spellings if(word[0] == entry[0])]
        
        dist = [(word, nltk.edit_distance(entry,word, transpositions=True)) for word in word_list]
           
        result.append(sorted(dist, key = lambda x: x[1])[0][0])
        
    return result
    
answer_eleven()


# In[ ]:



