# AdaBoost

### 简介

**AdaBoost**是一种提升方法，其基本做法是：在每一轮训练中，提高那些被前一轮弱分类器**错误分类**样本得权值，而降低那些被**正确分类**样本得权值。



### 推导（基于“加性模型”）

#### 模型假设

对于给定数据：

$$x\in D$$

$$f(x)\in\{-1, 1\}$$

假设，**AdaBoost**的**基学习器**为：

$$h_t(x)\ \ \ \ \ t\in\{1,2,\dots,T\} \tag{1}$$

其中，**第一个**基学习器$h_1$是通过直接将基学习器算法用于**初始数据分布**得到的，此后迭代地生成$h_t$和$\alpha_t$；

则，该算法的最终结果为，各基学习器的**线性组合**，即：

$$H(x)=\sum^T_{t=1}\alpha_th_t(x) \tag{2}$$

其中，$\alpha_t$ 为线性组合的**权重**；

且，训练的**损失函数**为：

$$l_{exp}(H|D)=\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H(x)}] \tag{*}$$

$$l_{exp}(H|D)=\sum_{x\in D}\mathcal{D}(x)e^{-f(x)H(x)}$$

$$l_{exp}(H|D)=\sum_{i=1}^{|D|}\mathcal{D}(x_i)(e^{-H(x_i)}\mathbb{I}(f(x_i)=1)+e^{H(x_i)}\mathbb{I}(f(x_i)=-1))$$

$$l_{exp}(H|D)=\sum_{x\in\mathcal{D}}e^{-H(x_i)}P(f(x_i)=1|x_i)+e^{H(x_i)}P(f(x_i)=-1|x_i)$$



#### 模型求解

为了求得<font color=red>(*)</font>中$l_{exp}(H|D)$的最小值，考虑将$l_{exp}(H|D)$对$H(x)$**求偏导**，得到：

$$\frac{\partial l_{exp}(H|D)}{\partial H(x)}=-e^{-H(x)}P(f(x)=1|x)+e^{H(x)}P(f(x)=-1|x) \tag{3}$$

令<font color=red>(3)</font>式为0，可得：

$$H(x)=\frac{1}{2}\ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)} \tag{4}$$

因此，有：

$$sign(H(x))=sign(\frac{1}{2}\ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)})$$

$$sign(H(x))=\left\{\begin{array}{**lr**} 1,\ \ \ \ P(f(x)=1|x)>P(f(x)=-1|x)\\-1,\ \ P(f(x)=1|x) <P(f(x)=-1|x)\end{array}   \right.  $$

$$sign(H(x))=argmax_{y\in\{-1,1\}}P(f(x)=y|x) \tag{5}$$



当，基学习器$h_t$基于分布$\mathcal{D}_t$产生后，该基分类器的权重$\alpha_t$应使得$\alpha_th_t$**最小化指数损失函数**（<font color=red>(*)</font>），即：

$$l_{exp}(\alpha_th_t|\mathcal{D}_t)=\mathbb{E}_{x\sim \mathcal{D}_t}[e^{-f(x)\alpha_th_t(x)}]$$

$$l_{exp}(\alpha_th_t|\mathcal{D}_t)=\mathbb{E}_{x\sim \mathcal{D}_t}[e^{-\alpha_t}\mathbb{I}(f(x)=h(x))+e^{\alpha_t}\mathbb{I}(f(x)\neq h_t(x))]$$

$$l_{exp}(\alpha_th_t|\mathcal{D}_t)=e^{-\alpha_t}P_{x\sim\mathcal{D}_t}(f(x)=h_t(x))+e^{\alpha_t}P_{x\sim\mathcal{D}_t}(f(x)\neq h_t(x))$$

$$l_{exp}(\alpha_th_t|\mathcal{D}_t)=e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t \tag{6}$$



其中，$\epsilon_t=P_{x\sim \mathcal{D}_t}(h_t(x)\neq f(x))$，为求损失函数的最小值，对<font color=red>(6)</font>求导，得到：

$$\frac{\partial l_{exp}(\alpha_t h_t|\mathcal{D}_t)}{\partial \alpha_t}=-e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t \tag{7}$$

令<font color=red>(7)</font>式为0，得到：

$$\alpha_t=\frac{1}{2}\ln(\frac{1-\epsilon_t}{\epsilon_t})\tag{8}$$



**AdaBoost**在获得$H_{t-1}$之后，需要对**样本分布**进行调整，使得下一轮的基学习器$h_t$能纠正$H_{t-1}$的错误($$H_t=H_{t-1}+\alpha_th_t$$)，即最小化**损失函数**：

$$l_{exp}(H_{t-1}+h_t|\mathcal{D})=\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)(H_{t-1}(x)+h_t(x))}]$$

$$l_{exp}(H_{t-1}+h_t|\mathcal{D})=\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}e^{-f(x)h_t(x)}] \tag{9}$$

又有：$f^2(x)=h_t^2(x)=1$，所以<font color=red>(9)</font>式，可以使用$e^{-f(x)h_t(x)}$的**泰勒展开式**近似：

$$l_{exp}(H_{t-1}+h_t|\mathcal{D})\approx \mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac{f^2(x)h_t^2(x)}{2})]$$

$$l_{exp}(H_{t-1}+h_t|\mathcal{D})\approx \mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}(1-f(x)h_t(x)+\frac{1}{2})] \tag{10}$$



于是，**理想的第t个基学习器**：

$$h_t(x)=argmin_hl_{exp}(H_{t-1}+h|\mathcal{D})$$

$$h_t(x)=argmin_h\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}(1-f(x)h(x)+\frac{1}{2})]$$

$$h_t(x)=argmax_h\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}f(x)h(x)] \tag{11}$$



**注意到**$\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}]$是一个常数，令：

$$\mathcal{D}_t(x)=\frac{\mathcal{D(x)}e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\sim \mathcal{D}}[e^{-f(x)H_{t-1}(x)}]} \tag{12}$$

于是，式<font color=red>(11)</font>可以转化为：

$$h_t(x)=argmax_h\mathbb{E}_{x\sim\mathcal{D}}[\frac{e^{-f(x)H_{t-1}(x)}}{\mathbb{E}_{x\sim\mathcal{D}}[e^{-f(x)H_{t-1}(x)}]}f(x)h(x)]$$

$$h_t(x)=argmax_h\mathbb{E}_{x\sim\mathcal{D}_t}[f(x)h(x)] \tag{13}$$



由 $f(x),h(x)\in \{-1,+1\}$：

$$f(x)h(x)=1-2\mathbb{I}(f(x)\neq h(x)) \tag{14}$$

所以**理想的基学习器**为：

$$h_t(x)=argmin_h \mathbb{E_{x\sim\mathcal{D}_t}}[\mathbb{I}(f(x)\neq h(x))]$$

由上，可以发现，**理想的基学习器**$h_t$将在分布$\mathcal{D}_t$下最小化分类误差。



下面推导$\mathcal{D}_t$的更新公式，由<font color=red>(12)</font>得：

$$\mathcal{D}_{t+1}(x)=\frac{\mathcal{D(x)}e^{-f(x)H_{t}(x)}}{\mathbb{E}_{x\sim \mathcal{D}}[e^{-f(x)H_{t}(x)}]}$$

$$\mathcal{D}_{t+1}(x)=\frac{\mathcal{D(x)}e^{-f(x)H_{t-1}(x)}e^{-f(x)\alpha_th_t(x)}}{\mathbb{E}_{x\sim \mathcal{D}}[e^{-f(x)H_{t}(x)}]}$$

$$\mathcal{D}_{t+1}(x)=\mathcal{D}_t(x)e^{-f(x)\alpha_th_t(x)}\frac{\mathbb{E_{x\sim\mathcal{D}}}[e^{-f(x)H_{t-1}(x)}]}{\mathbb{E_{x\sim\mathcal{D}}}[e^{-f(x)H_t(x)}]} \tag{15}$$





### 算法

*****

**输入**：训练数据集$T=\{(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)\}$，其中$x_i\in\mathcal{X}\subseteq\mathbb{R}^n，y_i\in\mathcal{Y}=\{-1,+1\}$

**输出**：最终得分类器$H(x)$

****

* 初始化训练数据的权值分布

    $$\mathcal{D}_1=(w_{11},w_{12},\dots,w_{1N}),w_{1i}=\frac{1}{N},i=1,2,\dots,N$$

* 对$m=1,2,\dots,M$

    * 使用具有权值分布$\mathcal{D}_m$的训练数据集学习，得到基本分类器

        $$H_m(x):\mathcal{X}\to\{-1, +1\}$$

    * 计算$H_m(x)$在训练数据集上的分类误差率

        $$e_m=P(H_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}\mathbb{I}(H_m(x_i)\neq y_i)$$

    * 计算$H_m(x)$的**系数**

        $$\alpha_m=\frac{1}{2}\ln\frac{1-e_m}{e_m}$$

    * **更新训练数据集的权值分布**

        $$\mathcal{D}_{m+1}=(w_{m+1,1},w_{m+1,2},\dots,w_{m+1,N})$$

        $$w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iH_m(x_i)),i=1,2,\dots,N$$

        其中，$Z_m$是**规范化因子**：

        $$Z_m=\sum_{i=1}^Nw_{mi}\exp(-\alpha_my_iH_m(x_i))$$

        它使得$\mathcal{D}_{m+1}$成为一个概率分布

* 构建基本分类器的线性组合

    $$f(x)=\sum_{m=1}^M\alpha_mH_m(x)$$

* 得到最终的分类器

    $$H(x)={\rm sign}(f(x))={\rm sign}(\sum_{m=1}^M\alpha_mH_m(x))$$

****



