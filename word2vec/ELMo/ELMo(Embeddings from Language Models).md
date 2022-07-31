# ELMo(Embeddings from Language Models)

## Embedding Models

### one-hot

![image-20200617225357200](C:\Users\WILL\AppData\Roaming\Typora\typora-user-images\image-20200617225357200.png)

### skip-gram

#### 简介

*Skip-gram*主要是给定input word预测上下文

#### 模型

**假设**：我们考虑**中性词**和**中性词邻近单词**之间的关系，而这个**邻近**我们用一个固定的“**距离n**”来表示，即：距离中心词不超过n个单词的所有单词为**邻近单词**。例如下图中 **n=2**，这样我们就考虑*into*附近的*problems*、*turning*、*banking*、*crises*这些单词为**邻近单词**。因为这些**邻近单词**与**中心词**距离近，因此我们认为**邻近单词**和**中心词**的相关性更高。

![image-20200617225550556](C:\Users\WILL\AppData\Roaming\Typora\typora-user-images\image-20200617225550556.png)

**模型**：

### GloVe

**假设**：设定一个窗口大小(window size)。窗口中心在语料中从左向右滑动。窗口的中心对应的词叫中心词(center word)，中心左右两边的窗口内的词叫语境词(context word)。

**符号声明**：

$X_{i,j}$：在整个语料库中，单词$i$和单词$j$共同出现在一个窗口中的次数。

$f(X_{i,j})$：权重函数，满足以下条件：

 1. $f(0)=0$；

 2. $f(x)$是**非递减**的；

 3. $f(x)<1$。

    常用的$f(x)$有：

    $$ f(x)=\left\{ \begin{aligned} (x/x_{max})^\alpha & if x<x_{max}\\ 1 & otherwise \end{aligned} \right. $$

**代价函数**：

$$J=\sum_{i,j}^N{f(X_{i,j})(v_i^Tv_j+b_i+b_j-log(X_{i,j}))^2}$$

### 模型的局限

1. 以往的一个词对应一个向量，是固定的。
2. 这些模型只考虑了局部的词之间的联系，但有时候远距离的词之间也有联系



## ELMo

### 简介

**ELMo**是基于**BiLSTM**的**embedding**模型

### 模型简介

![img](https://pic3.zhimg.com/80/v2-443008ce3b8560978240ad4c9cfb58ba_1440w.jpg)

#### 前向LSTM

在一个长度为**N**的序列中，前向模型计算的在给定$(t_1,t_2,...t_{k-1})$序列的情况下，整个句子的概率：

$$p(t_1,t_2,...,t_N)=\prod_{k=1}^Np(t_k|t_1,t_2,...,t_{k-1})$$

模型的目标即是求使得$p(t_1,t_2,...,t_N)$最大的序列。



#### 后向LSTM

$$p(t_1,t_2,...,t_N)=\prod_{k-1}^Np(t_k|t_{k-1},t_{k-2},...,t_N)$$

模型的目标即是求使得$p(t_1,t_2,...,t_N)$最大的序列。

#### biLM

**biLM**是**双向模型**，因此它要优化的函数为：

$$\sum_{k=1}^N(logp(t_k|t_1,...,t_{k-1};\Theta_x,\overrightarrow{\Theta_{LSTM}},\Theta_S ))+logp(t_k|t_{k+1},...,t_N;\Theta_x,\overleftarrow{\Theta_{LSTM}},\Theta_s)$$



#### ELMo

对于每个输入(token)，一个L层的**biLM**可以计算出**2L+1**个表征(representations)：

$$R_k=\{x_k^{LM},\overrightarrow{h_{k,j}^{LM}},\overleftarrow{h_{k,j}^{LM}}|j=1,...L\}$$



#### 模型结果

$$\mathbf{ELMo}_k^{task}=E(R_k;\Theta^{task})=\gamma^{task}\sum_{j=0}^Ls_j^{task}\mathbf{h}_{k,j}^{LM}$$

其中：

$s^{task}$：*softmax*标准化的权重

$\gamma^{task}$：缩放模型的*vector*，合适的值对于实际应用的过程十分重要



### 训练过程

### NLP任务的应用

给定一个已经训练好的**biLM**和**监督模型的NLP任务**

首先我们运行**biLM**并记录对于每个单词，所有层的**表征**。然后，我们让最终的任务模型学习这些**表征**的**线性组合**。

对于不使用**biLM**的**监督模型**。