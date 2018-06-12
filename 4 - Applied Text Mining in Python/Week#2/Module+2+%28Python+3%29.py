
# coding: utf-8

# # Module 2 (Python 3)

# ## Basic NLP Tasks with NLTK

# In[1]:

import nltk
from nltk.book import *


# ### Counting vocabulary of words

# In[3]:

text7


# In[11]:

sents()


# In[12]:

sent7


# In[13]:

len(sent7)


# In[14]:

len(text7)


# In[15]:

len(set(text7))


# In[16]:

list(set(text7))[:10]


# ### Frequency of words

# In[62]:

dist = FreqDist(text7)
len(dist)
dist


# In[66]:

import operator
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_x = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
sorted_x


# In[22]:

vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]


# In[26]:

dist['four']


# In[27]:

freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords


# ### Normalization and stemming

# In[28]:

input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
words1


# In[29]:

porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# ### Lemmatization

# In[30]:

udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]


# In[31]:

[porter.stem(t) for t in udhr[:20]] # Still Lemmatization


# In[33]:

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]


# ### Tokenization

# In[34]:

text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')


# In[35]:

nltk.word_tokenize(text11)


# In[40]:

text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
len(sentences)


# In[41]:

sentences


# ## Advanced NLP Tasks with NLTK

# ### POS tagging

# In[57]:

nltk.help.upenn_tagset('JJ')


# In[43]:

text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)


# In[52]:

text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)


# In[54]:

# Parsing sentence structure
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


# In[55]:

text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1


# In[41]:

parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)


# In[42]:

from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# ### POS tagging and parsing ambiguity

# In[43]:

text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)


# In[44]:

text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)


# In[58]:

string1 = "salam"
print(string1.upper())


# In[ ]:



