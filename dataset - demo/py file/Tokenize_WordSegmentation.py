#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
dataTraining = pandas.read_csv(filepath_or_buffer='vnDictionary.csv', 
                       usecols=['Input'])


# In[2]:


from tokenizer import tokenize, TOK
Tokens = [] 
for data in dataTraining['Input']:
    token = tokenize(data)
    Tokens.append(token) 


# In[3]:


for token in Tokens[0]:
    print("{0}: '{1}'".format(
        TOK.descr[token.kind],
        token.txt or "-"))


# In[4]:


import nltk # Sử dụng nltk lib để tách từ
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[5]:


from nltk.tokenize import word_tokenize

dataChanged = [] 
for data in dataTraining['Input'].fillna(""):
    token = word_tokenize(data)
    dataChanged.append(token) 
    
dataChanged[0]


# In[6]:


from pyvi import ViTokenizer, ViPosTagger

dataTraining['Input'] = dataTraining['Input'].fillna("").apply(ViTokenizer.tokenize)
dataTraining['Input'][0]


# In[7]:


dataChanged = []
for rowData in dataTraining['Input']:
    data = ViPosTagger.postagging(rowData)
    dataChanged.append(data)

print(pandas.DataFrame(dataChanged[:1]))


# In[ ]:




