import re, collections
from typing import Union
import json

class BPE(object):
    def __init__(self, num_merge=10) -> None:
        """
        @Param
        ------
        :num_merge 重复聚合的次数
        """
        super().__init__()
        self.num_merge = num_merge
    
    # @property
    # def num_merge(self):
    #     return num_merge

    def get_stats(vocab) -> collections.defaultdict:
        """
        获取每种pair出现的次数
        @Param
        ------
        :vocab(dict) 单词的字典[格式: 'v o c a b <\w>': 2]

        @Return
        -------
        :pair(dict) 每个单词对的统计次数[格式 ('v', 'o'): 3]
        """
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(pair, v_in) -> dict:
        """
        进行一次聚合操作即将句子的每个 'pair[0]  pair[1]' -> 'pair[0]pair[1]'
        @Param
        ------
        :pair(dict) 待聚合的pair
        :v_in(dict) 上一轮的vocab

        @Return
        :v_out(dict) 输出的vocab
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    def txt_to_vocab(path:str, encoding='utf-8'):
        """
        将文本转成vocab
        @Param
        ------
        :path(str) 存储路径
        @Return
        -------
        :vocab(dict) vocab数据
        """
        data = None
        with open(str, encoding=encoding) as file:
            data = file.readlines()
        vocab = collections.defaultdict(int)
        for line in data:
            for word in line.split():
                chars = ' '.join(word) + '<\w>'
                vocab[chars] += 1
        return vocab

    def train(self, data:Union[str, dict], save_path:str) -> None:
        """
        进行数据处理
        @Param
        ------
        :data(dict/str) 数据路径或者dict数据
        """
        if type(data) == str:
            data = BPE.txt_to_vocab(data)

        for i in range(self.num_merge):
            pairs = BPE.get_stats(data)
            if len(pairs) == 0:
                break
            pair = max(pairs, key=pairs.get)
            data = BPE.merge_vocab(pair, data)
        
        lines = set()
        for word in data:
            for char in word.split():
                lines.add(char)
        lines = list(lines)
        with open(save_path, encoding='utf-8', mode='w+') as file:
            file.write('\n'.join(lines))

# 频率词典
vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merge = 10
bpe = BPE(10)
bpe.train(vocab, 'test.txt')


