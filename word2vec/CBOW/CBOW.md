# CBOW（Continuous Bag-of-Words）

### 简介

**CBOW**在给定一个**窗口**的**上下文**时，预测**中心词**。



![](.\CBOW.png)



### 模型公式

***

$$P(w_i|w_{j(|j-i|\le l,j\neq i)})=Softmax(M(\sum_{|j-i|\le l,j\neq i}w_j))$$

**其中**

$P(w_i|w_{j(|j-i|\le l,j\neq i)})$：是在给定其**上下文**时，**中心词**为$w_i$的概率

$l$：是**窗口**大小

$M$：是**权值矩阵**，$M=\mathbb{R}^{|V|\times m}$

***

**CBOW**用以下**代价函数**做**最优化**

$$\mathscr{L}=-\sum_{i}\log P(w_i|w_{j(|j-i|\le l,j\neq i)})$$

$l$：是待调节的**超参数**，越大准确度越高，但训练代价越大。



### Hierarchical Softmax（层次softmax）

### Negative sampling（负采样）

