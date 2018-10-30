
# coding: utf-8

# In[10]:


import csv


# In[25]:


trainfile = open("train.csv", "r")
testfile = open("test.csv", "r")


# In[29]:


train = csv.DictReader(trainfile)
test = csv.DictReader(testfile)


# In[28]:




