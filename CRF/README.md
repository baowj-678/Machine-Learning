# 条件随机场（CRF）

### 简介：

设$X$与$Y$是**随机变量**，$P(Y|X)$是在给定$X$条件下$Y$的**条件概率分布**。如果**随机变量**$Y$构成一个由**无向图**$G=(V,E)$表示的**马尔可夫随机场**，即：

$$P(Y_v|X,Y_w,w \neq v)=P(Y_v|X,Y_w,w\sim v)$$

对任意结点$v$成立，则称**条件概率分布**$P(X|Y)$为**条件随机场**。

 $w \sim v$：表示在图$G=(V,E)$中与结点$v$有边连接的所有结点；

$w \neq v$：表示结点$v$以外的所有结点。

***

**线性条件随机场**：

设$X=(X_1, X_2,\cdots ,X_n),Y=(Y_1,Y_2,\cdots,Y_n)$均为线性链表示的随机变量序列，若在给定随机变量序列$X$的条件下，随机变量序列$Y$的**条件概率分布**$P(Y|X)$构成条件随机场，即满足**马尔可夫性**

$$P(Y_i|X,Y_1,\cdots,Y_{i-1},Y_{i+1},\cdots,Y_n)=P(Y_i|X,Y_{i-1},Y_{i+1})\quad i=1,2,\cdots,n$$

在标注问题中，$X$表示输入**观测序列**，$Y$表示对应输出**标记序列**或**状态序列**

（在**NER**问题中，$X$可以为**一句话**吗，$Y$为对每个词标注的**词性**）



### 模型：

 #### 线性条件随机场的参数化形式：

设$P(Y|X)$为**线性条件随机场**，则在随机变量$X$取值为$\vec{x}$的条件下，随机变量$Y$取值为$y$的**条件概率**具有如下形式：

$$P(y|x)=\frac{1}{Z(x)}\exp\Big(\sum_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\Big)$$

**其中**，

$$Z(x)=\sum_y\exp\Big(\sum_{i,k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{i,l}\mu_ls_l(y_i,x,i)\Big)$$

**其中**，

$t_k$：是**特征函数**，称为**转移特征**，依赖于当前和前一个位置；

$s_l$：是**特征函数**，称为**状态特征**，依赖于当前位置；

$\lambda_k,\mu_l$：是对应的权值；

$Z(x)$：是**规范化因子**；

（通常特征函数取值为$\{0,1\}$，但满足特征时取1，否则为0）



#### 线性条件随机场的简化矩阵形式：

**令**:

$y_0=start,y_{n+1}=stop$

$y_{1\to n}\in\{y^*_1,y^*_2,\cdots,y^*_m\}$

**有**：

$$W_i(y_{i-1},y_i|x)=\sum_{k}\lambda_kt_k(y_{i-1},y_i,x,i)+\sum_{l}\mu_ls_l(y_i,x,i)$$

$$M_i(y_{i-1},y_i|x)=\exp(W_i(y_{i-1},y_i|x))$$

$$M_i(x)=\begin{pmatrix} M_i(y^*_1,y^*_1|x) & M_i(y^*_1,y^*_2,|x) & \cdots & M_i(y^*_1,y^*_m|x) \\ M_i(y^*_2,y^*_1|x) & M_i(y^*_2,y^*_2,|x) & \cdots & M_i(y^*_2,y^*_m|x) \\ \vdots & \vdots & \ddots & \vdots \\ M_i(y^*_m,y^*_1|x) & M_i(y^*_m,y^*_2|x) & \cdots & M_i(y^*_m,y^*_m|x) \end{pmatrix}$$

**所以**：

$$P_w(y|x)=\frac{1}{Z_w(x)}\prod_{i=1}^{n+1}M_i(y_{i-1},y_i|x)$$

$$Z_w(x) = [M_1(x)M_2(x)\cdots M_{n+1}(x)]_{start,stop}$$



### 条件随机场的计算：

#### 前向-后向算法:

**前向向量**：

$$\alpha_i(y_i|x)$$：表示在位置$i$的标记是$y_i$ 并且从$1$到$i$的**前部分标记序列**的**非规范化概率**

$$\alpha_i(x)=\begin{pmatrix} \alpha_i(y_i=y^*_1|x) \\ \alpha_i(y_i=y^*_2|x) \\ \vdots \\ \alpha_i(y_i=y^*_m|x)\end{pmatrix}$$

**递推公式**：

$$\alpha_i^T(y_i|x)=\alpha_{i-1}^T(y_{i-1}|x)M_i(x)$$

$$\alpha_0(y|x)=\begin{cases} 1,\quad y=start \\ 0,\quad otherwise \end{cases}$$

****

**后向向量**：

$$\beta_i(y_i|x)$$：表示在位置$i$的标记是$y_i$ 并且从$i+1$到$n$的**前部分标记序列**的**非规范化概率**

$$\beta_i(x)=\begin{pmatrix} \beta_i(y_i=y^*_1|x) \\ \beta_i(y_i=y^*_2|x) \\ \vdots \\ \beta_i(y_i=y^*_m|x)\end{pmatrix}$$

**递推公式**：

$$\beta_i(y_i|x)=M_{i+1}(x)\beta_{i+1}(y_{i-1}|x)$$

$$\alpha_0(y|x)=\begin{cases} 1,\quad y=start \\ 0,\quad otherwise \end{cases}$$



#### 概率计算：

标记序列在位置$i$是$y_i$的**条件概率**为：

$$P(Y_i=y_i|x)=\frac{1}{Z(x)}\prod_{Y_0Y_1\cdots Y_{i-1}(Y_i=y_i)Y_{i+1}\cdots Y_{n+1}} M_i(x)$$

$$=\frac{\alpha_i^T(Y_i=y_i|x)\beta_i(Y_i=y_i|x)}{Z(x)}$$

***

标记序列在位置$i-1$、$i$是$y_{i-1}$、$y_i$的**条件概率**为：

$$P(Y_{i-1}=y_{i-1},Y_i=y_i|x)=\frac{1}{Z(x)}\prod_{Y_0Y_1\cdots (Y_{i-1}=y_{i-1})(Y_i=y_i)\cdots Y_{n+1}}M_i(x)$$

$$=\frac{\alpha_{i-1}^T(y_{i-1}|x)M_i(y_{i-1},y_i|x)\beta_i(y_i|x)}{Z(x)}$$



#### 期望计算：



