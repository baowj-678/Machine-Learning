# GloVe模型:boxing_glove:



## 预训练模型:relaxed:



### 预训练模型目录

| 地址                                                         | 详情                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [glove.6B](http://nlp.stanford.edu/data/glove.6B.zip)        | [Wikipedia 2014](http://dumps.wikimedia.org/enwiki/20140102/) + [Gigaword 5](https://catalog.ldc.upenn.edu/LDC2011T07) (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download):happy: |
| [glove.42B.300d](http://nlp.stanford.edu/data/glove.42B.300d.zip) | Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download):smiley: |
| [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) | Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):pensive: |
| [glove.twitter.27B](http://nlp.stanford.edu/data/glove.twitter.27B.zip) | Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download):cry: |



### 模型加载​（​gensim）:key:

**将原数据转成Word2Vec(gensim内部格式)**

~~~python
from scipy.stats.stats import mode
import torch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# 路径
glove_file = 'glove.6B.50d.txt'
tmp_file = "glove6B50d.txt"

# 将glove转成word2vec文件
_ = glove2word2vec(glove_file, tmp_file)
~~~



**加载模型(Word2Vec)**

~~~python
# 加载模型
model = KeyedVectors.load_word2vec_format(tmp_file)
~~~



**导入pytorch**:sun_with_face:

~~~python
# 初始化torch的Embedding
vocab_size = 10000
embed_size = 100
weight = torch.zeros(vocab_size + 1, embed_size)
for i in range(len(model.index2word)):
    try:
        # word_to_idx是vocab的函数，将单词映射到index
        index = word_to_idx[model.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(model.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

# 生成Embedding
embedding = torch.nn.Embedding.from_pretrained(weight)
~~~



## 项目目录

| 文件                        | 详情                     |
| --------------------------- | ------------------------ |
| [load_model](load_model.py) | **加载预训练模型**的代码 |

