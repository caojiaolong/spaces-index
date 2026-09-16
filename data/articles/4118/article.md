# 基于双向GRU和语言模型的视角情感分析

> 作者：苏剑林 · 科学空间 · 2016-12-01
>
> 原文：<https://spaces.ac.cn/archives/4118>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前段时间参加了一个傻逼的网络比赛——基于视角的领域情感分析，[主页在这里](<http://wid.org.cn/data/science/player/competition/detail/description/237>)。比赛的任务是找出一段话的实体然后判断情感，比如“我喜欢本田，我不喜欢丰田”这句话中，要标出“本田”和“丰田”，并且站在本田的角度，情感是积极的，站在丰田的角度，情感就是消极的。也就是说，等价于将实体识别和情感分析结合起来了。

### 吐槽

看起来很高端，哪里傻逼了？比赛任务本身还不错，值得研究，然而官方却很傻逼，主要体现为：1、比赛分初赛、复赛、决赛三个阶段，初赛一个多月时间，然后筛选部分进入复赛，复赛就简单换了一点数据，题目、数据的领域都没有变化，复赛也是一个月的时间，这傻逼复赛究竟有什么意义？2、大家可以看看选手们在群里讨论什么：

> 嗷嗷嗷嗷  17:40:54
> 128004	【杭州德奥奥迪品荐二手车】奥迪ttcoupe45tfsiquattro2015年53\.69万
> 嗷嗷嗷嗷  17:40:57
> @国双赛题指导
> 嗷嗷嗷嗷  17:41:09
> 这个视角取到什么位置啊
> 国双赛题指导  17:41:19
> 奥迪tt
> 
> 
> 
> 风云  20:19:47
> 没开过好车，感觉本田的操控比丰田 日产好吧     这里的“丰田”、“日产”应该neg还是neu
> 风云  20:20:00
> 感觉初赛复赛对这种标准不统一
> 风云  20:20:12
> @国双赛题指导 @国双赛题指导3
> 国双赛题指导  21:29:52
> neu
> 
> 
> 
> Kk\_asd  10:15:00
> @国双赛题指导 上海大众，上海要删掉吗？
> 国双赛题指导  10:15:18
> bu
> 
> 
> 
> 出门向右  20:49:06
> 有进口福特，这样的视角吗@国双赛题指导
> 出门向右  20:49:16
> 进口宝马？
> 国双赛题指导  20:54:43
> 没有
> 
> 
> 
> Kk\_asd  10:57:28
> 起亚律动出现了好多，要标出起亚吗？@国双赛题指导
> 国双赛题指导  11:43:04
> 不要

我也就不说什么了，如果官方认为这是机器学习，那就是机器学习吧，只是我看上去更像“管理员学习”。

反正是一个傻逼的比赛，我就也当一回傻逼吧。我也不奢望有什么名次，比赛还没结束，我先把我自己的模型公开了，大家如果成绩比我低的，可以按照这个模版，刷一下成绩。

### 模型

其实这个任务，我的做法跟[《基于双向LSTM和迁移学习的seq2seq核心实体识别》](<https://spaces.ac.cn/archives/3942/>)差不多，视为一个序列标注问题，只不过将LSTM换成了参数更少的GRU。这次我使用了字标注法，用0标注非实体部分，用1标注积极实体，用2标注中性实体，用3标注消极实体，仅此而已。由于标签语料是汽车领域的，我自己爬了一些汽车领域的语料，并且自己写了基于GRU的语言模型，用来训练字向量，因为我感觉Word2Vec的字向量做法太粗糙，对于小语料效果可能不好。

然后呢？没有然后了，剩下的基本就是重复[《基于双向LSTM和迁移学习的seq2seq核心实体识别》](<https://spaces.ac.cn/archives/3942/>)得了，连代码都一样。当然，最后它给出了一个汽车领域的实体列表，因此，我用这个列表在后期viterbi算法中进行了强行对齐。最后的迁移学习效果提升不大，大家看着办即可。

整个过程我自己比较满足的一点是端到端，语料下来后，几乎没有人工干预了。换个领域的语料，照样很快跑通。

### 效果

初赛准确率0\.56，复赛目前我的准确率0\.55，不算好，榜上最优成绩有0\.67的，不知道他们用什么方法做，希望有大神指导下。反正我是不打算做了。

### 代码



<pre><code class="language-python">#! -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import time
import os

print u'read data ...'
train_data = pd.read_csv('Train.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
test_data = pd.read_csv('Test.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
train_label = pd.read_csv('Label.csv', index_col='SentenceId', delimiter='\t', encoding='utf-8')
addition_data = pd.read_csv('addition_data.csv', header=None, encoding='utf-8')[0]
train_data.dropna(inplace=True) # drop some empty sentences
neg_data = pd.read_excel('neg.xls', header=None)[0]
pos_data = pd.read_excel('pos.xls', header=None)[0]

script_name = 'shibie.py'
now = int(time.time())
os.system('mkdir %s'%now)
os.system('cp %s %s'%(script_name, now))
os.system('cp addition_data.csv %s'%now)

# soma parameters
min_count = 5
maxlen = 100
word_size = 64

print u'making mapping dictionary ...'
word2id = ''.join(train_data['Content']) + ''.join(test_data['Content']) + ''.join(addition_data)
word2id = pd.Series(list(word2id)).value_counts()
word2id = word2id[word2id &gt;= min_count]
word2id[:] = range(1, len(word2id)+1)
print u'keep %s words.'%len(word2id)

def doc2id(s):
    return list(word2id[list(s)].fillna(len(word2id)+1).astype(np.int32))

print u'translating texts into id sequences ...'
train_data['doc2id'] = map(lambda i: doc2id(train_data.loc[i, 'Content']), tqdm(iter(train_data.index)))
test_data['doc2id'] = map(lambda i: doc2id(test_data.loc[i, 'Content']), tqdm(iter(test_data.index)))
addition_data[:] = map(lambda i: doc2id(addition_data[i]), tqdm(iter(addition_data.index)))
pos_data[:] = map(lambda i: doc2id(pos_data[i]), tqdm(iter(pos_data.index)))
neg_data[:] = map(lambda i: doc2id(neg_data[i]), tqdm(iter(neg_data.index)))

# make n-grams for train language model
n = 8
def gen_ngrams(s):
    s = [0]*(n-1) + s + [0]*(n-1)
    return zip(*[s[i:] for i in range(n)])

print u'generating ngrams ...'
from itertools import chain
ngrams = pd.concat([train_data['doc2id'].apply(gen_ngrams),
                    test_data['doc2id'].apply(gen_ngrams),
                    addition_data.apply(gen_ngrams),
                    pos_data.apply(gen_ngrams),
                    neg_data.apply(gen_ngrams)])
ngrams = np.array(list(chain(*ngrams)))

def findall(sub_string, string):
    start = 0
    idxs = []
    while True:
        idx = string[start:].find(sub_string)
        if idx == -1:
            return idxs
        else:
            idxs.append(start + idx)
            start += idx + len(sub_string)

tags = {'pos':1, 'neu':2, 'neg':3}

def label2tag(i):
    s = train_data.loc[i]['Content']
    r = np.array([0]*len(s))
    try:
        l = train_label.loc[[i]].as_matrix()
    except:
        return r
    for i in l:
        for j in findall(i[0], s):
            r[j:j+len(i[0])] = tags[i[1]]
    return r

print u'translating target into tags ...'
train_data['label'] = map(label2tag, tqdm(iter(train_data.index)))
print u'keep %s train sample.'%len(train_data)

from keras.layers import Input, Embedding, GRU, Dense, TimeDistributed, Bidirectional
from keras.models import Model
from keras.utils import np_utils

RNN = GRU # which type of RNN we used, try LSTM or GRU

# in order to gain good word embedding, we use GRU to train a n-grams language model
# it costs more time, but it produces better word embedding.
print u'training language model ...'
lm_input = Input(shape=(n-1,), dtype='int32')
lm_embedded = Embedding(len(word2id)+2,
                         word_size,
                         input_length=n-1,
                         mask_zero=True)(lm_input)
lm_rnn = RNN(64)(lm_embedded)
lm_output = Dense(len(word2id)+2, activation='softmax')(lm_rnn)
language_model = Model(input=lm_input, output=lm_output)
language_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def lm_generator(ngrams, batch_size):
    while True:
        np.random.shuffle(ngrams)
        for p in np.split(ngrams, range(batch_size, len(ngrams), batch_size)):
            yield p[:, :-1], np_utils.to_categorical(p[:, -1], len(word2id)+2)

nb_epoch = 8 # accuracy changes slightly after 5 epoch
batch_size = 4096
lm_history = language_model.fit_generator(lm_generator(ngrams, batch_size), nb_epoch=nb_epoch, samples_per_epoch=len(ngrams))
language_model.save_weights('%s/language_model_weights.model'%now)
structure = open('%s/language_model_structure.model'%now, 'w')
structure.write(language_model.to_json())
structure.close()

# here we use 2 layers of bidirectional GRU to make a sequence tagging model
print u'training ner model ...'
ner_input = Input(shape=(maxlen,), dtype='int32')
ner_embedded = Embedding(len(word2id)+2,
                         word_size,
                         input_length=maxlen,
                         mask_zero=True,
                         trainable=False,
                         weights=[language_model.get_weights()[0]])(ner_input)
ner_brnn = Bidirectional(RNN(64, return_sequences=True), merge_mode='sum')(ner_embedded)
ner_brnn = Bidirectional(RNN(32, return_sequences=True), merge_mode='sum')(ner_brnn)
ner_output = TimeDistributed(Dense(5, activation='softmax'))(ner_brnn)
ner_model = Model(input=ner_input, output=ner_output)
ner_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ner_data = train_data['doc2id'].apply(lambda s: s[:maxlen] + [0]*(maxlen - len(s[:maxlen])))
ner_data = np.array(list(ner_data))
ner_target = train_data['label'].apply(list).apply(lambda s: s[:maxlen] + [4]*(maxlen - len(s[:maxlen])))
ner_target = np.array(list(ner_target))
ner_target = np.array(map(lambda y:np_utils.to_categorical(y,5), ner_target))
sample_weight = (3/(train_data['label'].apply(lambda s:(np.array(s)==2).sum())+3)).as_matrix()

nb_epoch = 300
batch_size = 1024
ner_history_1 = ner_model.fit(ner_data, ner_target, batch_size=batch_size, nb_epoch=nb_epoch, sample_weight=sample_weight)
ner_model.save_weights('%s/ner_model_weights_1.model'%now)
structure = open('%s/ner_model_structure_1.model'%now, 'w')
structure.write(ner_model.to_json())
structure.close()

test_ner_data = test_data['doc2id'].apply(lambda s: s[:maxlen] + [0]*(maxlen - len(s[:maxlen])))
test_ner_data = np.array(list(test_ner_data))

print u'predicting ...'
train_data['predict'] = list(ner_model.predict(ner_data, batch_size=batch_size, verbose=1))
test_data['predict'] = list(ner_model.predict(test_ner_data, batch_size=batch_size, verbose=1))

def viterbi(nodes):
    paths = nodes[0]
    for l in range(1,len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1]+i in zy.keys():
                    nows[j+i]= paths_[j]+nodes[l][i]+zy[j[-1]+i]
            k = np.argmax(nows.values())
            paths[nows.keys()[k]] = nows.values()[k]
    return paths.keys()[np.argmax(paths.values())]

zy = {'00':1,
      '01':1,
      '02':1,
      '03':1,
      '10':1,
      '11':1,
      '20':1,
      '22':1,
      '30':1,
      '33':1}

zy = {i:np.log(zy[i]) for i in zy.keys()}

from acora import AcoraBuilder
views = pd.read_csv('View.csv', delimiter='\t', encoding='utf-8')['View']
views = AcoraBuilder(*views)
views = views.build()

def predict(i, data):
    y_pred = data.loc[i, 'predict']
    s = data.loc[i, 'Content'][:maxlen]
    nodes = [dict(zip(['0','1','2','3'], k)) for k in np.log(y_pred[:len(s)])]
    tags_pred_1 = viterbi(nodes)
    for j in views.finditer(s):
        for k in range(j[1], j[1]+len(j[0])):
            nodes[k]['1'] += 100
            nodes[k]['2'] += 100
            nodes[k]['3'] += 100
        try:
            nodes[j[1]-1]['0'] += 50
            nodes[k+1]['0'] += 50
        except:
            pass
    tags_pred_2 = viterbi(nodes)
    r = []
    for j in re.finditer('1+|2+|3+', tags_pred_2):
        t = pd.Series(list(tags_pred_1[j.start():j.end()])).value_counts()
        t = t[t.index != '0']
        if len(t) == 0:
            continue
        else:
            if t.index[0] == '1':
                r.append((i, s[j.start():j.end()], 'pos'))
            elif t.index[0] == '2':
                r.append((i, s[j.start():j.end()], 'neu'))
            else:
                r.append((i, s[j.start():j.end()], 'neg'))
    return r

print u'creating the final export ...'
train_data['pred'] = map(lambda i: predict(i, train_data), tqdm(iter(train_data.index)))
test_data['pred'] = map(lambda i: predict(i, test_data), tqdm(iter(test_data.index)))

result_1 = pd.DataFrame(list(chain(*test_data['pred'])), columns=['SentenceId', 'View', 'Opinion'])
result_1 = result_1.drop_duplicates()
result_1.to_csv('%s/result_1.csv'%now, index=None, encoding='utf-8')

# transfer learning
# we use the train result to train ner model again
result_1['SentenceId'] = result_1['SentenceId'].apply(int)
result = result_1.set_index('SentenceId')

def label2tag(i):
    s = test_data.loc[i]['Content']
    r = np.array([0]*len(s))
    try:
        l = result.loc[[i]].as_matrix()
    except:
        return r
    for i in l:
        for j in findall(i[0], s):
            r[j:j+len(i[0])] = tags[i[1]]
    return r

test_data['label'] = map(label2tag, tqdm(iter(test_data.index)))
ner_data = train_data['doc2id'].append(test_data['doc2id']).apply(lambda s: s[:maxlen] + [0]*(maxlen - len(s[:maxlen])))
ner_data = np.array(list(ner_data))
ner_target = train_data['label'].append(test_data['label']).apply(list).apply(lambda s: s[:maxlen] + [0]*(maxlen - len(s[:maxlen])))
ner_target = np.array(list(ner_target))
ner_target = np.array(map(lambda y:np_utils.to_categorical(y, 5), ner_target))

nb_epoch = 100
batch_size = 1024
ner_history_2 = ner_model.fit(ner_data, ner_target, batch_size=batch_size, nb_epoch=nb_epoch)
ner_model.save_weights('%s/ner_model_weights_2.model'%now)
structure = open('%s/ner_model_structure_2.model'%now, 'w')
structure.write(ner_model.to_json())
structure.close()

print u'predicting again ...'
test_data['predict'] = list(ner_model.predict(test_ner_data, batch_size=batch_size, verbose=1))

print u'creating the final export again ...'
test_data['pred'] = map(lambda i: predict(i, test_data), tqdm(iter(test_data.index)))

result_2 = pd.DataFrame(list(chain(*test_data['pred'])), columns=['SentenceId', 'View', 'Opinion'])
result_2 = result_2.drop_duplicates()
result_2.to_csv('%s/result_2.csv'%now, index=None, encoding='utf-8')</code></pre>



打包下载：[基于视角的领域感情分析\_打包\.7z](<https://spaces.ac.cn/usr/uploads/2016/12/234474787.7z>)
