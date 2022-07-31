# BLEU模型



### 简介

****

**BLEU**是评价**机器翻译**好坏的一种模型，给定**机器翻译**的结果和**人工翻译**的**参考译文**，该模型会自动给出翻译的得分，分数越高则表明翻译的结果越好。



### 模型建立过程

为了评价**翻译系统**(MT)的翻译结果的好坏，我们先观察好的翻译结果之间的**联系**，如下例子：

**Example 1**：

~~~
Candidate 1: It is a guide to action which ensures that the military always obeys the commands of the party.

Candidate 2: It is to insure the troops forever hearing the activity guidebook that party direct.

Reference 1: It is a guide to action that ensures that the military will forever
heed Party commands.

Reference 2: It is the guiding principle which guarantees the military forces
always being under the command of the Party.

Reference 3: It is the practical guide for the army always to heed the directions of the party.
~~~

***

### n-grams

从上面例子中我们可以发现**候选句子(Candidate)**和**参考句子(Reference)**之间存在一些相同的**句子片段**，例如：

**Candidate 1** - **Reference 1**： *"It is a guide to action"*， *"ensures that the military"*， *"commands"* 

**Candidate 1** - **Reference 2**：*"which"* ， *"always"* ， *"of the party"* 

**Candidate 1** - **Reference 3**：*"always"* 

而

**Candidate 2**和**参考句子**之间相同的**片段**数量不多。

于是，我们可以找到**候选句子**的每个**n-gram**在**参考句子**中出现的**次数**，并**求和**，通过上面的分析可以发现，匹配的个数越多，求和的值越大，则说明**候选句子**更好。



***

### Modified n-grams

**Example 2**：

~~~
Candidate: the the the the the the the.

Reference 1: The cat is on the mat.

Reference 2: There is a cat on the mat.
~~~

考虑上面的**Example 2**，如果按照**n-grams**的方法，**候选句子(Candidate)**，的每个**1-gram** *the*在**参考句子**中出现的次数都很多，因此，求和的得分也到，按照**n-grams**的评价方法，**Candidate**是个很好的翻译结果，但事实却**并非如此**。

于是，我们可以考虑修改**n-gram**模型：

1. 首先，计算一个单词在**任意**一个**参考句子**出现的最大次数；

2. 然后，用每个（非重复）**单词**在**参考句子**中出现的最大次数来**修剪**，单词在**候选句子**的出现次数；

   $$Count_{clip}=min(Count, Max\_Ref\_Count)$$

3. 最后，将这些**修剪**的次数加起来，除以总的**候选句子**词数。

例如在**Example 2**中：

1. *the*在**Ref 1**出现的次数为：2，在**Ref 2**出现的次数为：1；
2. *the*的**修剪**后的次数为：2；
3. *the*的最终值为**2/7**。

在**Example 1**中：

**Candidate 1**的得分为：**17/18**；

**Candidate 2**的得分为：**8/14**



***

### Modified n-grams on blocks of text

当我们在**长文本**中评价时：

1. 首先，**逐句**地计算**n-gram**匹配个数；
2. 然后，将所有**候选句子**的$Count_{clip}$加在一起，除以**测试语料库**中的**候选句子** **n-gram**总数，得到整个**测试语料库**的分数$p_n$



$$p_n=\frac{\sum_{C\in\{Candidates\}}\sum_{n-gram\in C}Count_{clip}(n-gram)}{\sum_{C'\in\{Candidates\}}\sum_{n-gram'\in C'}Count(n-gram')}$$

**其中**

$Candidates$：表示**机器翻译的译文**

$Count()$：在**Candidates**中**n-gram**出现的次数

$Count_{clip}()$：**Candidates**中**n-gram**在**Reference**中出现的次数



***

### Sentence length

**n-gram**惩罚**候选句子**中的不出现在**参考句子**中的单词；

**modified n-gram**惩罚在**候选句子**中比**参考句子**中出现次数多的单词；

***

### BLEU



$$BP=\left\{ \begin{aligned}1 \qquad if\quad c>r \\ e^{1-\frac{r}{c}} \qquad if \quad c\leq r \end{aligned}\right.$$

$c$：**Candidate语料库**的长度

$r$：**effective Reference**的长度：**Ref**中和**Candidate**中每句匹配的句子长度之和

****

$$BLEU=BP\cdot exp(\sum_{n=1}^Nw_n logp_n)$$

**取对数**后为：

$$\log BLEU=min(1-\frac{r}{c}, 0)+\sum_{n=1}^Nw_n\log P_n$$

$w_n$：权重，一般为$\frac1N$



****



## Code



~~~python
''' BLEU (BiLingual Evaluation Understudy)
@Author: Bao Wenjie
@Date: 2020/9/16
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
~~~



#### Reference：

[*Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a method for automatic evaluation of machine translation.* ](*https://doi.org/10.3115/1073083.1073135*)

[github](https://github.com/baowj-678/NLP/tree/master/BLEU)