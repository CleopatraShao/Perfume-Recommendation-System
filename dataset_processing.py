#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# In[2]:


reviews = pd.read_excel("reviews808K.xlsx")


# In[3]:


reviews.head()#查看前五项，用于判断读入是否准确


# In[4]:


blob = TextBlob(reviews['text'].iloc[0])
blob.tags


# In[5]:


#TextBlob是一个用Python编写的文本处理库，可以用于执行很多nlp任务，例如词性标注、情感分析等等，这里用其进行情感分析
testimonial = TextBlob("Textblob is amazingly simple to use!")#创建一个textblob对象


# In[6]:


for sentence in blob.sentences:
    print(sentence, sentence.sentiment.polarity)


# In[7]:


blob.sentiment


# In[8]:


def get_sentiment(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity


# In[9]:


reviews['text'] = reviews['text'].apply(lambda x: str(x))


# In[10]:


senti_rev = reviews['text'].apply(lambda x: get_sentiment(x))


# In[11]:


senti_rev.shape#查看数据量检验是否正确


# In[12]:


reviews['sentiment'] = senti_rev


# In[13]:


reviews.head()#查看前五项，判断输出是否正确


# In[14]:


reviews_modi = reviews


# In[15]:


reviews_modi['sentiment'] = reviews_copy['sentiment'] + 1
reviews_modi['sentiment'] = reviews_copy['sentiment'] / 2


# In[17]:


reviews.to_csv("reviews_senti.csv", index=None)
reviews_copy.to_csv("reviews_senti_modify.csv", index=None)

