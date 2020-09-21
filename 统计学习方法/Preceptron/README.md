# 感知机

### 简介

**感知机(preceptron)**是*二分类*的**线性分类**模型，其*输入*为实例的**特征向量**，*输出*为实例的**类别**（-1， 1）。



### 模型

**定义**：输入空间：$\mathcal{X}\subseteq\mathbb{R}^n$，输出空间：$\mathcal{Y}=\{+1,-1\}$，输入实例：$x\in\mathcal{X}$，输出的实例类别：$y\in\mathcal{Y}$，则：

$$f(x) = sign(\bold{w}\cdot x+b)$$

称为**感知机**。

$\bold{w}\in\mathbb{R}^n$：权值（weight）

$b\in\mathbb{R}^n$：偏置（bias）

$sign(x)=\left\{\begin{aligned} +1\quad x\geq 0 \\ -1\quad x<0\end{aligned}\right.$：符号函数



### 学习方法

#### 损失函数

输入空间$\mathbb{R}^n$任一点$x_0$到，**超平面**$\bold{S}$的距离：

$$D=\frac{1}{||\bold{w}||}|\bold{w}\cdot x_0+b|$$

对于**误分类**数据：

$$Loss=-\frac{1}{||\bold{w}||}y_i(\bold{w}\cdot x_i + b)>0$$

考虑**误分类**点到**超平面**的**总距离**，于是有**损失函数**：

$$Loss(\bold{w},b)=-\sum_{x_i\in M}y_i(\bold{w}\cdot x_i+b)$$



***

#### 迭代法

$$\nabla_{\bold{w}}L(\bold{w}, b)=-\sum_{x_i\in M}y_ix_i$$

$$\nabla_{b}L(\bold{w}, b)=-\sum_{x_i\in M}y_i$$

**算法**：

1. 选取初值$\bold{w}_0,b_0$

2. 在训练集选取数据$(x_i,y_i)$

3. 梯度更新：

   $$\left\{\begin{aligned} \bold{w}=\bold{w}+\eta\nabla_{\bold{w}}L(\bold{w}, b)\\ b=b+\eta\nabla_{b}L(\bold{w}, b)\end{aligned}\right.$$

