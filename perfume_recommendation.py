#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import re
import keras
from keras.layers import *
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, Embedding
from keras.layers import Concatenate, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.utils.vis_utils import model_to_dot
from keras import losses
from keras import optimizers
from keras.models import load_model
import gc
from IPython.display import SVG
from itertools import product


# In[2]:


#导入数据
cust_df = pd.read_csv("profilesniche_prepared.csv")
perf_df = pd.read_csv("products_finals_with_accords.csv")
reviews_df = pd.read_csv("reviews_senti_modify.csv")


# In[3]:


cust_df.shape#用于查看数据规模


# In[4]:


cust_df.head()#查看前五行的数据情况


# In[5]:


cust_df = cust_df[['IDcustomer', 'text']]


# In[6]:


cust_df.isna().sum()


# In[7]:


cust_df['text'].fillna("unknown", inplace = True)


# In[8]:


perf_df.shape


# In[9]:


perf_df.head()


# In[10]:


perf_df[['0', '1', '2', '3']]#对列进行命名，方便后续遍历删除


# In[11]:


#removeing names in front of nodes
del_strs = ['Top0', 'Top1', 'Top2', 'Top3', 'Middle0', 'Middle1', 'Middle2']
for i in ['0', '1', '2', '3']:
    for del_str in del_strs:
        perf_df[i] = perf_df[i].apply(lambda x: x.replace(del_str,''))


# In[12]:


perf_df['0'].nunique(), perf_df['1'].nunique(), perf_df['2'].nunique(), perf_df['3'].nunique(), 


# In[13]:


perf_df[['0', '1', '2', '3']].head()


# In[14]:


perf_df[['url', '0', '1', '2', '3']].to_csv("perfume_nodes.csv", index=None)


# In[15]:


dummy = list(perf_df['0']) + list(perf_df['1']) +  list(perf_df['2']) +  list(perf_df['3']) 


# In[16]:


dummies = set(dummy)


# In[17]:


len(dummies)


# In[18]:


nodes_encoding = {i:j for j, i in enumerate(dummies)}


# In[19]:


nodes_encoding['Cetalox']#举一个例子看看是否正确


# In[20]:


perf_df['Top0'] =  perf_df['0'].apply(lambda x:nodes_encoding[x])
perf_df['Top1'] =  perf_df['1'].apply(lambda x:nodes_encoding[x])
perf_df['Top2'] =  perf_df['2'].apply(lambda x:nodes_encoding[x])
perf_df['Top3'] =  perf_df['3'].apply(lambda x:nodes_encoding[x])

del perf_df['0']
del perf_df['1']
del perf_df['2']
del perf_df['3']


# In[21]:


perf_df.head()


# In[22]:


del perf_df['title']


# In[23]:


perf_df.head()


# In[25]:


reviews_df.head()


# In[26]:


for col in ['avatar_img', 'date', 'karma', 'name_perfume', 'rew', 'username']:
    del reviews_df[col]


# In[27]:


reviews_df.head()


# In[28]:


def get_user_id(x):
    vals = re.findall(r'\d+', x)
    return vals[0]

reviews_df['IDcustomer'] = reviews_df['userlink'].apply(lambda x: get_user_id(x))


# In[29]:


del reviews_df['userlink']


# In[30]:


reviews_df.head()


# In[31]:


cust_df.head()


# In[32]:


perf_df.head()


# In[33]:


url_encoding = {}
for i, j in enumerate(perf_df['url']):
    url_encoding[j] = i


# In[34]:


perf_df['ID_perfume'] = perf_df['url'].apply(lambda x:url_encoding[x])


# In[35]:


perf_df.head()


# In[36]:


reviews_df.head()


# In[37]:


gc.collect()


# In[38]:


def get_url_encoding(x):
    try:
        vals = url_encoding[x]        
    except:
        vals = None
    return vals


# In[39]:


reviews_df['ID_perfume'] = reviews_df['url'].apply(lambda x: get_url_encoding(x))


# In[40]:


reviews_df.tail()


# In[41]:


reviews_df.isna().sum()


# In[42]:


reviews_df = reviews_df[ratings_df['ID_perfume'].isna()== False]


# In[43]:


reviews_df.shape


# In[44]:


del reviews_df['url']
del perf_df['url']
del reviews_df['text']


# In[45]:


cust_df.head()


# In[46]:


perf_df.head()


# In[47]:


reviews_df.head()


# In[48]:


reviews_df['ID_perfume'] = reviews_df['ID_perfume'].astype('int')


# In[49]:


cust_df.shape, perf_df.shape, reviews_df.shape


# In[50]:


reviews_df['IDcustomer'].dtypes, cust_df['IDcustomer'].dtypes 


# In[51]:


reviews_df['IDcustomer'] = reviews_df['IDcustomer'].astype('int')


# In[52]:


reviews_df.head()


# In[53]:


del reviews_df['text']


# In[54]:


existing_df = pd.merge(reviews_df, cust_df, on='IDcustomer', how='left')


# In[55]:


existing_df = pd.merge(existing_df, perf_df, on='ID_perfume', how='left')


# In[56]:


existing_df.shape


# In[57]:


existing_df.head()


# In[58]:


existing_df.isna().sum()


# In[59]:


existing_df['text'].fillna('unknown', inplace = True)


# In[60]:


existing_df.head()


# In[61]:


combination_existed = existing_df[['IDcustomer', 'ID_perfume']]


# In[62]:


dummy = existing_df['text'].apply(lambda x :len(x.split()))


# In[63]:


dummy.describe()


# In[64]:


def clean_text(x):
    x = x.lower()
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    return x


# In[65]:


existing_df['text'] = existing_df['text'].apply(lambda x:clean_text(x))


# In[66]:


gc.collect()


# In[67]:


embed_size = 300 # word vector的大小
max_features = 100000 # how many unique words to use
maxlen = 150 # max number of words in a question


# In[68]:


my_list = [i for i in existing_df['text']]


# In[69]:


my_words = " ".join(i for i in my_list)


# In[70]:


words_length = len(set(my_words.split()))
words_length


# In[71]:


max_features = words_length


# In[ ]:





# In[ ]:





# In[72]:


## fill up the missing values
train_X = existing_df['text'].fillna("##").values

print("Before Tokenization:")
print(train_X.shape)


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

print("After Tokenization:")
print(len(train_X))


# In[73]:


train_X = pad_sequences(train_X, maxlen=maxlen)


# In[74]:


gc.collect()


# In[75]:


existing_df.head()


# In[76]:


x_train = existing_df.drop(['sentiment', 'IDcustomer',  'ID_perfume', 'text'], axis=1)


# In[77]:


x_train.shape


# In[78]:


train_X.shape


# In[79]:


import numpy as geek 


# In[81]:


gc.collect()


# In[82]:


x_train.shape, train_X.shape, existing_df['sentiment'].shape


# In[83]:


np.random.seed(0)
indices = np.random.permutation(x_train.shape[0])
#为了使得划分更为随机，采用上述两行代码增加随机性
training_idx, test_idx = indices[:700000], indices[700000:]#划分训练集和测试集


# In[84]:


x_train, x_test = x_train.iloc[training_idx,:], x_train.iloc[test_idx,:]

x_train_embed, x_test_embed = train_X[training_idx,:], train_X[test_idx,:]


# In[85]:


x_train.shape, x_test.shape


# In[86]:


target = existing_df['sentiment'].values


# In[87]:


target


# In[88]:


target_encoded = np.where(target>0.6, 1, 0)


# In[89]:


target_encoded


# In[90]:


y_train, y_test = target_encoded[training_idx], target_encoded[test_idx]


# In[91]:


y_train.shape, y_test.shape


# In[92]:


gc.collect()


# In[93]:


x_train_embed.shape, x_train.shape


# In[94]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)

x = LSTM(256, return_sequences=True)(x)
x = LSTM(64, return_sequences=True)(x)
x = Flatten()(x)

agei = Input(shape=(153,))

conc = concatenate([x, agei])

drop = Dropout(0.2)(conc)
dens = Dense(100)(drop)
dens = Dense(1)(dens)
acti = Activation('sigmoid')(dens)

model = Model(inputs=[inp, agei], outputs=acti)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['acc'])


# In[96]:


model.fit([x_train_embed, x_train], y_train, validation_data=([x_test_embed, x_test], y_test),epochs=1,
          batch_size=1024, shuffle=True, verbose=1)


# In[103]:


gc.collect()


# In[105]:


gc.collect()


# In[106]:


my_list = list(product(existing_df['IDcustomer'][:10], existing_df['ID_perfume'].unique()))
newdf = pd.DataFrame(data=my_list, columns=['IDcustomer','ID_perfume'])


# In[107]:


newdf['IDcustomer'].nunique()


# In[108]:


newdf['IDcustomer']


# In[109]:


newdf['combo'] = newdf['IDcustomer'].apply(str) + " " + newdf['ID_perfume'].apply(str)
combination_existed['combo'] = combination_existed['IDcustomer'].apply(str) + " " + combination_existed['ID_perfume'].apply(str)


# In[110]:


combination_existed['combo'].values


# In[111]:


test_df = newdf.loc[~newdf['combo'].isin(combination_existed['combo'].values)]


# In[112]:


del test_df['combo']


# In[113]:


gc.collect()


# In[114]:


del existing_df
del newdf
del combination_existed


# In[115]:


gc.collect()


# In[116]:


test_df = pd.merge(test_df, cust_df, on='IDcustomer', how='left')
test_df = pd.merge(test_df, perf_df, on='ID_perfume', how='left')


# In[117]:


test_df.shape


# In[118]:


test_X = test_df['text'].fillna("##").values
test_X = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(test_X, maxlen=maxlen)


# In[119]:


test_df.head()


# In[120]:


x_test = test_df.drop(['IDcustomer',  'ID_perfume', 'text'], axis=1)


# In[121]:


x_test.shape


# In[122]:


test_X.shape


# In[123]:


y_pred = model.predict([test_X, x_test])


# In[124]:


y_pred[:5]


# In[125]:


test_df['prediction'] = y_pred


# In[126]:


test_df.head()


# In[127]:


def get_recommendations(cust_id):
    results = test_df[test_df['IDcustomer']==cust_id][['ID_perfume', 'prediction']]
    return results.sort_values(by ='prediction' , ascending=False)[:10]


# In[128]:


get_recommendations(1141379)

