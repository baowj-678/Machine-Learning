import pandas as pd

class Vocab(object):
    def __init__(self, path):
        self.word2int = {}
        self.unk_id = 0
        self.load_dict(path)

    def __len__(self):
        return len(self.word2int)
    
    def __getitem__(self, word):
        return self.word2int.get(word, self.unk_id)
    
    def __contains__(self, word):
        return (word in self.word2int)
    
    def load_dict(self, path):
        data = pd.read_csv(path)
        print('--'*8 + '开始加载字典' + '--'*8)
        i = 1
        for ind, word in data.iterrows():
            self.word2int[word['Phrase']] = i
            i += 1
        print('--'*8 + '字典加载完毕' + '--'*8)
        

if __name__ == '__main__':
    vocab = Vocab('data\\vocab.tsv')
