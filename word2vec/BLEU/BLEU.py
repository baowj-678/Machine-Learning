''' BLEU (BiLingual Evaluation Understudy)
@Author: Bao Wenjie
@Date: 2020/8/2
@Email: bwj_678@qq.com
'''
import numpy as np

class BLEU():
    def __init__(self, n_gram=1):
        super().__init__()
        self.n_gram = n_gram

    def evaluate(self, candidates, references):
        ''' 计算BLEU值
        @param candidates [[str]]: 机器翻译的句子
        @param references [[str]]: 参考的句子
        @param bleu: BLEU值
        '''

        BP = 1
        bleu = np.zeros(len(candidates))
        for k, candidate in enumerate(candidates):
            r, c = 0, 0
            count = np.zeros(self.n_gram)
            count_clip = np.zeros(self.n_gram)
            count_index = np.zeros(self.n_gram)
            p = np.zeros(self.n_gram)
            for j, candidate_sent in enumerate(candidate):
                # 对每个句子遍历
                for i in range(self.n_gram):
                    count_, n_grams = self.extractNgram(candidate_sent, i + 1)
                    count[i] += count_
                    reference_sents = []
                    reference_sents = [reference[j] for reference in references]
                    count_clip_, count_index_ = self.countClip(reference_sents, i + 1, n_grams)
                    count_clip[i] += count_clip_
                    c += len(candidate_sent)
                    r += len(reference_sents[count_index_])
                p = count_clip / count
            rc = r / c
            if rc >= 1:
                BP = np.exp(1 - rc)
            else:
                rc = 1
            p[p == 0] = 1e-100
            p = np.log(p)
            bleu[k] = BP * np.exp(np.average(p))
        return bleu
            

    def extractNgram(self, candidate, n):
        ''' 抽取出n-gram
        @param candidate: [str]: 机器翻译的句子
        @param n int: n-garm值
        @return count int: n-garm个数
        @return n_grams set(): n-grams 
        '''
        count = 0
        n_grams = set()
        if(len(candidate) - n + 1 > 0):
            count += len(candidate) - n + 1
        for i in range(len(candidate) - n + 1):
            n_gram = ' '.join(candidate[i:i+n])
            n_grams.add(n_gram)
        return (count, n_grams)
    
    def countClip(self, references, n, n_gram):
        ''' 计数references中最多有多少n_grams
        @param references [[str]]: 参考译文
        @param n int: n-gram的值s
        @param n_gram set(): n-grams

        @return:
        @count: 出现的次数
        @index: 最多出现次数的句子所在文本的编号
        '''
        max_count = 0
        index = 0
        for j, reference in enumerate(references):
            count = 0
            for i in range(len(reference) - n + 1):
                if(' '.join(reference[i:i+n]) in n_gram):
                    count += 1
            if max_count < count:
                max_count = count
                index = j
        return (max_count, index)


if __name__ == '__main__':
    bleu_ = BLEU(4)
    candidates = [['It is a guide to action which ensures that the military always obeys the commands of the party'],
                 ['It is to insure the troops forever hearing the activity guidebook that party direct'],
    ]
    candidates = [[s.split() for s in candidate] for candidate in candidates]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands'],
                  ['It is the guiding principle which guarantees the military forces always being under the command of the Party'],
                  ['It is the practical guide for the army always to heed the directions of the party']
    ]
    references = [[s.split() for s in reference] for reference in references]
    print(bleu_.evaluate(candidates, references))