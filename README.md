# Perfume-Recommendation-System

### 选题动机

香水距今已有五百多年的发展史。在这漫长的历史过程中，调香师们通过对不同香料的选取和调配使其绽放出独一无二、与众不同的香味。然而，这也给人们带来了甜蜜的负担。

由于香水种类繁多，个人的了解深度和广度总是有限的，即使是在互联网的帮助下，也难以从千馥万香中寻觅到自己的最爱。因此，人们的挑选方式往往是前往实体商场直接进行试香，这不仅耗费了顾客的大量时间，同时也使得许多小众品牌因为缺乏曝光渠道而无人问津。因此我设计了“香你所想”香水推荐生成器，希望顾客能够通过自然语言的描述，让机器在短时间内反馈最为推荐的几款香水(可由顾客自行设置数量)，从而提升客户体验，同时也给中小企业更多的曝光机会。

从实际生活出发，大多数人对于香水并没有十分专业的了解，因此无法针对香料、前中后调性等提出精确的要求。可以说，大多数情况下顾客提出的只是一个模糊的要求，例如希望能够在办公室等工作场合中使用、希望闻起来有夏日的清新感等等，也有很多情况下，顾客会选择描述自己的年龄、性别、职位以及使用场合，希望能够基于这些条件获得推荐，这便需要本香水推荐生成器能够挖掘出其中的隐含信息。此外，由于个人的叙述语言以及风格不同，香水推荐生成器也需要将顾客的大量描述统一考虑在内，这对于神经网络模型的选取也是一大挑战。

在本项目中，我通过在目前全球最大的香水网站上爬取80万条香水的购买记录（其中包括香水的香料组成、前中后调性、品牌以及购买者对于这一香水的评价以及对自身的一些描述），采用LSTM神经网络模型对70万条数据进行分类学习以及训练，并将剩余10万条数据作为测试集进行测试。最终能够在30分钟左右完成模型训练，并给出香水推荐。

### 环境配置

python 3.8.5

其余可直接通过requirements.txt进行import

### 数据获取及处理

##### **基础数据集一：**

https://www.kaggle.com/ajithumer/parisdata?select=rewiews808K.xlsx

本数据集共计包含808k条数据，爬取数据来源为全球最大的香水网站www.fragrantica.com，其核心内容为顾客对于其所购买的香水的评价以及个人主页上对自己的描述。**该网站的优点在于其结合了购物(类似淘宝)以及个人评价(类似小红书)为一体，用户会在个人主页上对自身情况、对于香水的审美喜爱进行描述，同时会对自己购买的香水进行购物评价**，因此可以通过爬取这两项语料提升模型的准确程度，也为后续将其中部分数据作为测试集提供便利。评价语言为英文，词数分布在10-800词，以20-500词为主要分布区间。以下给出几例：

```
I blind bought Aka’ula which took me two weeks to figure out if I liked it as it was challenging in the dry down. I even roped my friends in to test it over night - and every single one like me said it was great for the first few hours but the next morning they said - nope - has an awkward unusual drydown. This is despite the fact that I have a tolerant nose for awkward scents.,
OneBlueSummers review below is clearly meant for this and posted here by accident I am assuming with all his references to lava... ,
However... it really impressed me with the quality of ingredients and technical performance.,
So I bought this and monto’ac...,
This is of the same build and blows me away...,
Amazingly despite the heavy coniferous and mint dose placing it in close competition to House of Sillage 003, Slumberhouse Grev, Oliver Durbano Jade and Turquoise and Heeley Menthe Fraiche this easily justifies a purchase to sit amongst them. Yes, I also had Diptyque Eau De Minthe and it is not in the same league IMHO BTW. Unlike these guys the ridiculously cheap packaging seems to infer that they are happy to invest the money in the quality... I have no problem with that...,
Great projection and 3-5 hours before it’s a waft from a trail.,
Despite the note table I would describe it reminding me also of fresh crushed sweet parsley and wormwood.,
Well done Source Adage! I can only hope you expand your range quickly...I look forward from Aka’ula to your next release...
```

```
I worked for Dior during the launch of this formula of MDC, in 2011. I recall it being quite the fanfare and we all got a big 100ml bottle. The first formula was much more gourmand it had caramelized popcorn and strawberry sorbet notes which made it very sweet and girly. It was intended for a younger market. This version was concocted to be more grown up, a bit sharper and more appealing to all ages. I do prefer it to the original and the current formulation. ,
This is a semi sweet jasmine fruitchouli perfume. The bottle is heavy and well constructed. Longevity is above average and it has good projection. It was reformulated slightly a few years ago when Dior decided to drop the cherie from the name. Then again last year for whatever reason. ,
If you get your hands on this one your lucky, it's not really my style but it is well made and very pretty on the right skin on the right day. I decided to spray some this morning because a coworker of mine wears this and it inspired me to give it a second chance. I will hold on to my bottle, the patchouli saves this for me, might add it to my spring rotation and see how that goes.
```

```
I really love this scent. It's one of those that you spray on your wrists and you can't quote stop smelling yourself. I begged my husband to buy it for me as he wasn't too impressed with it in the store. Well on me, he basically has the same reaction... He thinks it's ok but too powdery.
```

在本项目中，首先需要针对数据集中的香水名称、顾客评价两大类别进行分类处理和情感分析。

1.读入数据

```python
reviews = pd.read_excel("reviews808K.xlsx")
```

2.针对评价，即`text`列进行词性标注

```python
blob = TextBlob(reviews['text'].iloc[0])
blob.tags
```

3.TextBlob作为一个用Python编写的文本处理库，可以执行很多NLP任务，例如词性标注、情感分析、文本分类(Naive Bayes, Decision Tree)等等。这里主要运用其情感分析`(sentiment analysis)`的功能。`sentiment`的返回结果为一个元组`Sentiment(polarity, subjectivity)`，其中`polarity score`是`[-1.0, 1.0]`之间的浮点数，`-1.0`表示消极，`1.0`表示积极，而`subjectivity`是`[0.0,1.0]`之间的浮点数，`0.0`表示客观，`1.0`表示主观。

```python
#首先，创建一个textblob对象
testimonial = TextBlob("Textblob is amazingly simple to use!")
#随后调用.sentiment
for sentence in blob.sentences:
    print(sentence, sentence.sentiment.polarity)
```

因此通过上述方式可以得出每条评论的情感倾向以及客观程度。为了方便后续操作，定义函数`get_sentiment`，随后得出sentiment值，并且将其添加在表格的`sentiment`列中。

```python
def get_sentiment(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity
reviews['text'] = reviews['text'].apply(lambda x: str(x))
senti_rev = reviews['text'].apply(lambda x: get_sentiment(x))
reviews['sentiment'] = senti_rev
```

4.发现所得到的数值普遍比较小，为了方便后续的处理，同时保证`sentiment`的结果范围仍然在`[0.0,1.0]`之间，因此采用`将结果+1随后/2`的方式对数值进行整体放大：

```python
reviews_modi = reviews
reviews_modi['sentiment'] = reviews_copy['sentiment'] + 1
reviews_modi['sentiment'] = reviews_copy['sentiment'] / 2
```

5.最后将处理后的结果生成`.csv`文件输出

```python
reviews.to_csv("reviews_senti.csv", index=None)
reviews_copy.to_csv("reviews__senti_modify.csv", index=None)
```



##### **基础数据集二：**

https://www.kaggle.com/ajithumer/parisprodaccords

本数据集为网站www.fragrantica.com上所有香水的基本信息，包含香水名、持香度、调性(例如果香调、花香调、木质调等等)、生成香料(1表示含有这种香料，0表示不含有这种香料)等等。



##### **基础数据集三：**

https://www.kaggle.com/ajithumer/parisdata?select=profilesniche_prepared.csv

本数据集共有42484条数据，其中核心内容为各位顾客的ID号以及其在本网站(www.fragrantica.com)上购买过的、标注过`like`的、收藏入收藏夹`favourite`的、标注过`dislike`的各种香水，以及采用自然语言描述的对于香水的期望。例如

```
购买过的(have):
/perfume/Burberry/London-for-Men-804.html,
/perfume/Carolina-Herrera/Herrera-For-Men-289.html,
/perfume/Syed-Junaid-Alam/Hajar-22397.html,
/perfume/Yves-Saint-Laurent/La-Nuit-de-l-Homme-5521.html,
/perfume/Clinique/Clinique-Happy-373.html

标注过like的：
/perfume/Kenzo/L-Eau-par-Kenzo-pour-Homme-79.html,/perfume/Issey-Miyake/L-Eau-d-Issey-Pour-Homme-Intense-1998.html,/perfume/Zara-Home/Cuir-Velvet-42694.html,/perfume/Givenchy/Givenchy-pour-Homme-Blue-Label-38.html,/perfume/Calvin-Klein/Eternity-for-Men-Summer-2016-35912.html

收藏入favourite的：
/perfume/Cartier/Declaration-307.html,
/perfume/Christian-Dior/Sauvage-31861.html,
/perfume/Parfum-d-Empire/Tabac-Tabou-32091.html,
/perfume/Olfactive-Studio/Panorama-29903.html,
/perfume/Byredo/M-Mink-10758.html,/perfume/Andrea-Maack/Craft-10647.html

标注过dislike的：
/perfume/Axe/Dark-Temptation-29897.html,
/perfume/Olivier-Durbano/Lapis-Lazuli-41040.html,
/perfume/Parfums-Quartana/Poppy-Soma-39235.html
```

然而，本语料的难点在于顾客的描述并不精确，她们不会明确表明自己希望获得哪个品牌、哪个调性、哪个价格区间或者是由哪些香料组成的香水。相反地，她们通常会描述自身的审美看法，或者是对某些香水的思考等等，更偏向于一种自身观点的表达。为了更清晰地表现出本语料的特点，下面举出两个例子:

在本例中顾客详细地给出了自己偏好的香料

```
Im a feminine  sweet and fun person. Im creative glamorous down to earth  enthusiastic and Im in love with life! And that s because I love what I do: design magazines and newspapers for a living. Im such a perfumista since my parents (both) tough me to love fragrances since I was a little child  because they are such a perfumistas  full time. I love vanilla sweet fragrances  but Im opened to try new aromas. I wish I could work for a fashion house  and learn more about this word  I would LOVE to be a full time nose. Perfume is one of my passions. I love gourmand smells  such as honey  caramel  praline  milky notes  sugary notes  almonds  chocolate  coconut  and so on. My relationship with white flowers isn t good at all.  Lily of the valley is not well appreciated in my scents  and so it is violet. My favorite flowers are honeysuckle  orchids  neroli  gardenia  and orange blossom. Im not a fruity scented girl  and fruity florals are not my very favorites  but It depends how well its mixed and how complex the structure is. Pear  apple  and tropical notes are not well received in my collection.
```

然而在实际生活中，人们往往难以对香水有如此专业的理解，提出如此准确的要求。更多情况下，顾客提出的是一个模糊的对于香水的想法，例如希望闻起来具有少女气息、希望香气淡雅但是持久等等。还有一些情况下，香水推荐更为困难——人们有时会单纯描述自己的年龄、性别、职业以及使用场合和自己的一些审美想法，随后希望得到香水的推荐。这种描述对于目前的自然语言处理很有难度，因为需要挖掘出其中的隐含信息。例如，一位女大学生提出自己即将参与前台销售的面试，希望得到香水的推荐。这便需要模型能够挖掘出相关的信息。这里给出一例：

```
 I adore all things feminine and womanly. I believe a lady should be a lady  perfume accentuates the femininity. I m a label snob  I love fashion  animals  vintage cars and the New York Knicks.As a 31-year old woman from Germany,I love strong perfumes with a lot of character  that tell a story or put you in a special mood. I have a thing for rose scents. My haves are only full size bottles never minis or samples. I will never leave the house without mascara and I believe a little red lipstick can change your mood.
```



### 算法选择

#### 前期数据处理

以下对代码中的重要部分以及步骤进行详细的解释和说明，省略了一些输出检查的重复步骤。

1. 导入数据

   ```python
   cust_df = pd.read_csv("profilesniche_prepared.csv")
   perf_df = pd.read_csv("products_finals_with_accords.csv")
   reviews_df = pd.read_csv("reviews_senti_modify.csv")
   ```

2. 在本实验中统一采用`.shape`查看数据量的大小，采用`.head`或者`.tail`查看前五行或者后五行的数据来判断数据处理后的输出是否正确，例如

   ```python
   cust_df.shape#用于查看数据规模，根据输出可知共有66130行数据，12列
   ```

   ```python
   cust_df.head()#查看前五行的数据情况
   ```

3. 由于数据集中有空缺值，因此首先调用`isna()`寻找并查看共有多少缺失

   ```python
   cust_df.isna().sum()
   ```

   随后调用pandas中的`fillna()`进行填充，方便之后的处理

   ```python
   cust_df['text'].fillna("unknown", inplace = True)
   ```

4. 由于香水分为前中后多调，获得的数据集中采用`Top_x(x=0~2)`和`Middle_x(x=0~2)`进行区分。然而在香水推荐中一种香味属于前调、中调还是后调的影响是有限的，且涉及的语料中也没有对其进行重点强调，甚至如果考虑在内还会增加后续模型的处理时间，因此在这里我选择将表格中的这几个字符串删除，具体代码为

   ```python
   perf_df[['0', '1', '2', '3']]#首先对列进行命名，方便后续通过for循环进行删除
   ```

   随后将需要删除的字符串放入一个数组中，方便判断

   ```python
   #removeing these strings in front of nodes
   del_strs = ['Top0', 'Top1', 'Top2', 'Top3', 'Middle0', 'Middle1', 'Middle2']
   for i in ['0', '1', '2', '3']:
       for del_str in del_strs:
           perf_df[i] = perf_df[i].apply(lambda x: x.replace(del_str,''))
   ```

   为了验证已经成功删除，通过`perf_df[['0', '1', '2', '3']].head()`查看


5. 随后将处理过的数据重新写入表格中

   ```python
   perf_df[['url', '0', '1', '2', '3']].to_csv("perfume_nodes.csv", index=None)
   dummy = list(perf_df['0']) + list(perf_df['1']) +  list(perf_df['2']) +  list(perf_df['3']) 
   dummies = set(dummy)
   ```

   数据修改完毕后，将列名重新返还给原有列，需要注意的是这里应该采用`del`删除其与数据之间的链接，而不是直接删除数据

   ```python
   perf_df['Top0'] =  perf_df['0'].apply(lambda x:nodes_encoding[x])
   perf_df['Top1'] =  perf_df['1'].apply(lambda x:nodes_encoding[x])
   perf_df['Top2'] =  perf_df['2'].apply(lambda x:nodes_encoding[x])
   perf_df['Top3'] =  perf_df['3'].apply(lambda x:nodes_encoding[x])
   
   del perf_df['0']
   del perf_df['1']
   del perf_df['2']
   del perf_df['3']
   ```

6. 随后便是对`reviews_df`以及`cust_df`的表格进行一系列数据处理，主要工作为

   （1）寻找表格内的空缺值并采用`fillna()`进行填充

   （2）采用`merge`进行一系列数据合并

   （3）采用`astype`以及`apply(str)`进行数据转换，方便后续传入模型

   以上代码都在文件中给出了注释，且各个代码的输出结果也已经放在`perfume_recommendation.ipynb`中，故这里不再赘述。

7. 由于起初在CPU上运行，发现经常出现内存不足的情况，因此在代码中我多次使用了`gc.collect()`函数，其作用是进行内存回收，返回值为释放掉的对象个数。

   

#### 神经网络模型选择

本次项目中，我选择了LSTM作为核心的神经网络模型。

##### **选择原因**

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。也就是说相比普通的RNN，LSTM能够在更长的序列中有更好的表现。而本项目中提供的语料恰为需要考虑全局化信息的长序列，因此自然而然地想到选择LSTM。

与其他神经网络模型相比，LSTM在以下三方面具有优势，而这恰恰符合本项目的语料：

1. 属于时序数据（有时间依赖性），并且要求**全局化处理**；
2. 输入和输出的元素级别的对应上可能有不小的时间跨度；
3. 数据适中，不太短但也不会太长

##### 具体实现

1.将NLP语料转换成单词索引序列，主要通过`Tokenizer`实现

首先用`Tokenizer`中的`fit_on_texts`方法学习出语料的字典

```python
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
```

word_index 是对应的单词和数字的映射关系dict，通过dict可以将每个string的每个词转成数字，可以用texts_to_sequences。

```python
train_X = tokenizer.texts_to_sequences(train_X)
```

详细代码为

```python
## 填充空缺值
train_X = existing_df['text'].fillna("##").values

print("Before Tokenization:")
print(train_X.shape)

## 对sentences进行Tokenize
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

print("After Tokenization:")
print(len(train_X))
```



2.采用padding的方法将上述结果填充成相同的长度

```python
train_X = pad_sequences(train_X, maxlen=maxlen)
```

这样处理结束后便可以用keras中的embedding进行向量化，用于输入到LSTM中

```python
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
```



3.划分训练集和测试集，由于爬取有一定的顺序，因此为了使得训练集以及测试集的数据更为随机，采用`random.permutation()`以及`random.seed()`

```python
np.random.seed(0)

indices = np.random.permutation(x_train.shape[0])
training_idx, test_idx = indices[:700000], indices[700000:]
```

```python
x_train, x_test = x_train.iloc[training_idx,:], x_train.iloc[test_idx,:]

x_train_embed, x_test_embed = train_X[training_idx,:], train_X[test_idx,:]
```

最终训练集和测试机的数据量为：训练集700,000条，测试集103,218条



4.接下来正式使用LSTM进行训练。起初我采用了最直接的LSTM方法，即直接调用keras中的LSTM代码，但效果欠佳。在查找大量文献后，我在一篇博客(https://blog.csdn.net/ssswill/article/details/88533623)中发现其中提到如果采用flatten()会大幅提升效果。这是因为`Flatten`层能够用于将输入“压平”，即把多维的输入一维化，常用于从卷积层到全连接层的过渡。重要的是，`Flatten`不会影响batch的大小。

因此我向其中加入了`flatten()`函数

```
x = Flatten()(x)
```



5.神经网络在其输入和输出层之间具有隐藏层，这些隐藏层中嵌入了神经元，神经元内的权重以及神经元之间的连接使得神经网络系统能够模拟学习过程。神经网络体系结构中的神经元和层越多，其表示能力就越强，而表示能力的提高意味着神经网络可以拟合更复杂的函数，并可以更好地泛化到训练数据。

然而，越深的神经网络越容易出现过度拟合的问题，即模型在训练数据上表现良好，但经过训练的机器学习模型无法很好地泛化到不看见的数据。

而Dropout的主要目的是使网络中过度拟合的影响最小化。它通过随机减少神经网络中相互连接的神经元的数量来实现。在每一个训练步骤中，每个神经元都有可能被排除在外（从连接的神经元中被剔除）。在某种意义上，层内的神经元学习的权重值不是基于其相邻神经元的协作。

```
drop = Dropout(0.2)(conc)
```



6.Dense层，即全连接层，其实现的操作为：`output = activation(dot(input, kernel) + bias)` 其中 `activation` 是按逐个元素计算的激活函数，`kernel` 是由网络层创建的权值矩阵，以及 bias 是其创建的偏置向量 (只在 use_bias=True 时才有用)。

```python
dens = Dense(100)(drop)
dens = Dense(1)(dens)
```



7.采用sigmoid函数作为激活函数

```
acti = Activation('sigmoid')(dens)
```

损失函数则采用默认设置

```python
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['acc'])
```

完整代码为

```python
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
```

##### 运行时间

平均时间为38分钟，全代码运行平均时间为43分钟。

#### 优化思考

在论文《R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS（2017）》中提到，`We choose to use Gated Recurrent Unit(GRU)(Cho et al.,2014) in our experiment since it performs similarly to LSTM(Hochreiter & Schmidhuber, 1997) but is computationally cheaper`，因此在本实验中如果采用GRU代替LSTM，能够在不损失模型性能的情况下减少计算时间。

下简单介绍为何GRU能够对LSTM进行计算性能的提升。

首先，GRU将LSTM的cell state以及activation state视为一个state，即GRU只有一个state。也就是说，GRU中只需要控制一个状态，其cell state是等于activation state的，而LSTM中控制两个状态，cell state需要经过output gate后才能得到activation state。

此外，LSTM有3个gate而GRU只有2个gate，因此结构相对更加简单，需要训练的参数也更少，因此实现时间会更短。

从公式来看，GRU的两个gate分别为reset gate和update gate。

reset gate是对t-1时刻状态重置，它是一个经过sigmoid激活的输出。


随后reset gate用来做 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D+) 的计算：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D%3D%5Ctanh%28W_c%5B%5CGamma_r%5Ctimes+c%5E%7B%3Ct-1%3E%7D%2Cx%5E%7B%3Ct%3E%7D%5D%2Bb_c%29)

从公式可以看出，当前时刻的候选状态 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D) 并不是完全将 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct-1%3E%7D) 用来学习，而是要先reset一下，而在LSTM中，计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D+) 时候直接就用 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bc%7D%3D%5Ctanh%28W_c%5Ba%5E%7B%3Ct-1%3E%7D%2C+x%5E%7B%3Ct%3E%7D%5D%2Bb_c%29) 。

另外，GRU将update gate既当作update用，又当作forget用。

![[公式]](https://www.zhihu.com/equation?tex=c%5E%7B%3Ct%3E%7D%3D%5CGamma_u%5Ctimes+%5Ctilde%7Bc%7D%5E%7B%3Ct%3E%7D%2B%281-%5CGamma_u%29%5Ctimes+c%5E%7B%3Ct-1%3E%7D)

其中 ![[公式]](https://www.zhihu.com/equation?tex=1-%5CGamma_u) 就相当于LSTM中的forget gate。而LSTM中的forget gate是独立的，与update gate没有很大的关联，因此在这里GRU相当于对LSTM进行了合理的简化，从而减少了一定计算量，但是在性能上与LSTM没有区别。

由于GRU也在keras库中，因此只需要

```python
from keras.layers import GRU
```

随后对代码(LSTM修改为GRU)和数据输入格式进行一些极其简单的微调即可。

LSTM也有优势之处。根据《On the Practical Computational Power of Finite Precision RNNs for Language Recognition》，LSTM的cell state是无界的，可以模拟计数器。模拟的方式为将input gate 置 1，让全部新信息进入；forget gate 置为 1，让过去的信息全部保留；于是 cell state 就可以每次加 1（减 1 的情况也类似）。当用RNN识别形式语言 ![[公式]](https://www.zhihu.com/equation?tex=a%5Enb%5En) 和 ![[公式]](https://www.zhihu.com/equation?tex=a%5Enb%5Enc%5En) ，可以在 LSTM 的 cell 中找到统计字符个数的单元()，而 GRU 由于隐状态有界很难做到这一点。


### 结论和感想

在本次项目中，我通过对80万条香水的基本信息、顾客的购买评价以及顾客的自身情况介绍，采用一系列数据清洗及处理方法、LSTM等一系列神经网络模型进行学习训练，最终完成了“香你所想”香水推荐生成器。这是我第一次采用机器学习的方法完成NLP项目，真正体验到了从0到1的过程，收获颇丰。在此期间，我查阅了大量文献和博客，从一个简单的小task入手，学习数据清洗及处理的方法、常见库的函数、各类模型的原理、应用场景、优缺点以及如何使用，就像一块海绵投入海洋，尽自己所能不断地汲取丰富的知识。然而海洋实在太过浩渺，我本希望能够搭建一个前端网站，真正创造出可落地的推荐器，但限于时间最终没能完全实现。现阶段的生成器需要在excel表格中自行添加需求，随后运行代码才能够获得推荐。

### 参考文献

> Gail Weiss, Yoav Goldberg, Eran Yahav《On the Practical Computational Power of Finite Precision RNNs for Language Recognition》(2018)

> Microsoft Asia Natural Language Computing Group.《R-NET: MACHINE READING COMPREHENSION WITH SELF-MATCHING NETWORKS》(2017)

> 《Understanding LSTM Networks》http://colah.github.io/posts/2015-08-Understanding-LSTMs/

> 《LSTM原理及实现》bill_b https://blog.csdn.net/weixin_44162104/article/details/88660003

> Keras中文文档 https://keras.io/zh/
