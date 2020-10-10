#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
dataTraining = pandas.read_csv("spam.csv", encoding = "latin-1")
dataTraining = dataTraining[['v1','v2']]
dataTraining = dataTraining.rename(columns = {'v1': 'Label', 'v2': 'Content'})
dataTraining['Label'].value_counts()


# In[2]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import stem
import re
import random

stemmer = stem.SnowballStemmer('english')
stopWords = set(stopwords.words('english'))

def replace_words(text): 
    LatinChar = '[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]'
    SpecialAndSpaceChar = '[/^$*+?#!@{}&\n\t\f\r\.\,\)\(]'
    #Xử lý lọc dữ liệu
    deleteLatin = re.sub(LatinChar, '', text).strip()
    text = re.sub(SpecialAndSpaceChar, '', deleteLatin).strip()
    return text

def wordTokenize(dataSet):
    Tokens = []
    for data in dataSet:
        Tokens.append(word_tokenize(data))
    return Tokens

def extractWords(dataSet):
    dataWordSet = []
    index = 0
    Tokens = wordTokenize(dataSet)   
    while index < len(Tokens):
        tmp = ""
        for word in Tokens[index]:
            if stemmer.stem(word) not in stopWords:
                tmp = tmp + stemmer.stem(word) + ' '
            
        dataWordSet.append(tmp.strip())                
        index+=1
    return dataWordSet

def word_DataSet_Sum(dataSet):
    index = 0
    wordSum = 0
    while index < len(dataSet):
        wordSum = wordSum + len(dataSet[index])
        index+=1
        
    return wordSum

def count_DuplicateWord(data, dataSet):
    countSet = []
    for word in data:
        count = 0
        index = 0
        while index < len(dataSet):       
            count = count + dataSet[index].count(word)
            index+=1    
        countSet.append(count)
    return countSet

def P_dataOnDataSet(countSet, word_DataSet_Sum):
    result = 1
    for num in range(len(countSet)):
        result = result * (countSet[num] / word_DataSet_Sum)
        
    return result

def TypeOfData(C_ham, C_spam):
    if max(C_ham, C_spam) == C_ham:
        print("Type Of Data is: Ham")
    else:
        print("Type Of Data is: Spam")
    


# In[3]:


P_ham = (dataTraining['Label'].values == 'ham').sum() / len(dataTraining['Label'])
print("P(C_ham) = ", P_ham)
P_spam = 1 - P_ham
print("P(C_spam) = ", P_spam)

dataHamSet = extractWords(dataTraining[dataTraining['Label'] == 'ham']['Content'].apply(replace_words))
dataSpamSet = extractWords(dataTraining[dataTraining['Label'] == 'spam']['Content'].apply(replace_words))

testSet = dataHamSet + dataSpamSet
dataTest = word_tokenize(random.choice(testSet))

word_hamData_Sum = word_DataSet_Sum(wordTokenize(dataHamSet))
print("word hamData Sum = ", word_hamData_Sum)
word_spamData_Sum = word_DataSet_Sum(wordTokenize(dataSpamSet))
print("word spamData Sum", word_spamData_Sum)

countDataHamSet = count_DuplicateWord(dataTest, wordTokenize(dataHamSet))
print(countDataHamSet)
countDataSpamSet = count_DuplicateWord(dataTest, wordTokenize(dataSpamSet))
print(countDataSpamSet)

P_dataOnHam = P_dataOnDataSet(countDataHamSet, word_hamData_Sum) 
P_dataOnSpam = P_dataOnDataSet(countDataSpamSet, word_spamData_Sum) 
C_ham = P_dataOnHam * P_ham
C_spam = P_dataOnSpam * P_spam
print("P_dataOnHam = ", P_dataOnHam)
print("P_dataOnSpam = ", P_dataOnSpam)
print("C*_ham = ", C_ham)
print("C*_spam = ", C_spam)
TypeOfData(C_ham, C_spam)

