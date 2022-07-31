# Skip-Gram

### 简介

**Skip-gram**在给定**中心词**的情况下，预测其**上下文**单词。

![skip-gram](.\skip-gram.png)



### 模型公式

***

$$P(w_j|w_i)=Softmax(\pmb{Mw}_i)(|j-i|\le l,j\neq i)$$

**其中**

$P(w_j|w_i)$：是给定**中心词**，其**上下文**是$w_j$的概率。

$\pmb M$：是**权值矩阵**

***

**Skip-gram**模型损失函数和**CBOW**相似

$$\mathscr{L}=-\sum_i\sum_{j(|j-i|\le l,j \neq i)}P(w_j|w_i)$$

