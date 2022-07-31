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

# 加载模型
model = KeyedVectors.load_word2vec_format(tmp_file)

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
