''' one-hot representation
@Author: Bao Wenjie
@Date: 2020/8/3
@Email: bwj_678@qq.com
'''
from vocab import Vocab
import numpy as np


class OneHot():
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
    
    def embedding(self, word):
        embedded = np.zeros((1, len(self.vocab)))
        embedded[0, self.vocab[word]] = 1
        return embedded
