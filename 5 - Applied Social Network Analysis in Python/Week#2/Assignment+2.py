
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[1]:

import networkx as nx

# This line must be commented out when submitting to the autograder
#!head email_network.txt


# ### Question 1
# 
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[7]:

def answer_one():
    
    emailG = nx.read_edgelist('email_network.txt', data=[('time', int)],  create_using=nx.MultiDiGraph())
    
    return emailG
answer_one()


# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[8]:

def answer_two():
        
    emailG = answer_one()
    
    return (nx.number_of_nodes(emailG), nx.number_of_edges(emailG))
answer_two()


# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[15]:

def answer_three():
    
    emailG = answer_one()
    
    return (nx.is_strongly_connected(emailG), nx.is_weakly_connected(emailG))
answer_three()


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[21]:

def answer_four():
        
    emailG = answer_one()
    largest_weakly_connectivity  = sorted(nx.weakly_connected_components(emailG))
    
    return len(largest_weakly_connectivity[0])
answer_four()


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[40]:

def answer_five():
    
    emailG = answer_one()    
    largest_strongly_connectivity  = sorted((nx.strongly_connected_components(emailG)))
    
    return len(max(largest_strongly_connectivity, key = len))
answer_five()


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[45]:

def answer_six():
        
    emailG = answer_one()
    
    G_sc = max(nx.strongly_connected_component_subgraphs(emailG), key=len)
    
    return G_sc
answer_six()


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[47]:

def answer_seven():
        
        
    return nx.average_shortest_path_length(answer_six())
answer_seven()


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[48]:

def answer_eight():
        
    
    return nx.diameter(answer_six())
answer_eight()


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[49]:

def answer_nine():
           
    return set(nx.periphery(answer_six()))
answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[52]:

def answer_ten():
        
    # Your Code Here
    
    return set(nx.center(answer_six()))
answer_ten()


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[121]:

import numpy as np
def answer_eleven():
    
    G_sc =  answer_six()   
    dia = nx.diameter(G_sc)
    periph_list = list(set(nx.periphery(G_sc)))
    result = []
    for node in periph_list:
        p = nx.shortest_path_length(G_sc, node).values()
        
        result.append((node,len([val for val in p if val == dia])))
    
    return max(result, key = lambda x : x[1])
answer_eleven()


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[129]:

def answer_twelve():
        
    G_sc =  answer_six()
    center_list = list(set(nx.center(G_sc)))
    my_set = set([])
    for node in center_list:
               
        my_set = my_set.union(nx.minimum_node_cut(G_sc, node, "97"))
    
    return len(my_set)
answer_twelve()


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[132]:

def answer_thirteen():
    
    G_sc =  answer_six()
    G_un = nx.Graph(G_sc.to_undirected())
    
    return G_un
answer_thirteen()


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[134]:

def answer_fourteen():
        
    G_un = answer_thirteen()
    
    return (nx.transitivity(G_un), nx.average_clustering(G_un))
answer_fourteen()

