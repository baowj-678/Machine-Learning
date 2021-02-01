# LiBSVM:deciduous_tree:

### 简介

LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM[模式识别](https://baike.baidu.com/item/模式识别/295301)与回归的软件包，他不但提供了[编译](https://baike.baidu.com/item/编译/1258343)好的可在Windows系列系统的[执行](https://baike.baidu.com/item/执行/3012)文件，还提供了[源代码](https://baike.baidu.com/item/源代码/3969)，方便改进、修改以及在其它操作系统上应用；该软件对SVM所涉及的参数调节相对比较少，提供了很多的[默认参数](https://baike.baidu.com/item/默认参数/9856139)，利用这些默认参数可以解决很多问题；并提供了交互检验(Cross Validation)的功能。该软件可以解决C-SVM、ν-SVM、ε-SVR和ν-SVR等问题，包括基于一对一算法的多类[模式识别问题](https://baike.baidu.com/item/模式识别问题/22062727)。(via [百度百科](https://baike.baidu.com/item/LIBSVM/10483771?fr=aladdin))



### LiSVM数据集

**LiSVM**数据集是十分著名的**机器学习数据集**

#### 数据格式

该数据集使用如下方法编码数据：

~~~
[label] [index1]:[value1] [index2]:[value2] ...
[label] [index1]:[value1] [index2]:[value2] ...
~~~

其中：

**label**：数据标签；

**index**：数据维度（按递增顺序）；

**value**：数据值；

下面是原[数据](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t)

~~~
8 1:88 2:92 3:2 4:99 5:16 6:66 7:94 8:37 9:70 12:24 13:42 14:65 15:100 16:100
8 1:80 2:100 3:18 4:98 5:60 6:66 7:100 8:29 9:42 12:23 13:42 14:61 15:56 16:98
8 2:94 3:9 4:57 5:20 6:19 7:7 9:20 10:36 11:70 12:68 13:100 14:100 15:18 16:92
9 1:95 2:82 3:71 4:100 5:27 6:77 7:77 8:73 9:100 10:80 11:93 12:42 13:56 14:13
9 1:68 2:100 3:6 4:88 5:47 6:75 7:87 8:82 9:85 10:56 11:100 12:29 13:75 14:6
1 1:70 2:100 3:100 4:97 5:70 6:81 7:45 8:65 9:30 10:49 11:20 12:33 14:16
~~~



### 安装(python下)

**使用pip安装**

~~~python
pip install libsvm
~~~



### 简单使用

~~~python
from libsvm.svmutil import svm_train, svm_save_model, svm_predict
from libsvm.commonutil import svm_read_problem

train_data = svm_read_problem('dataset\ijcnn1\ijcnn1')
test_data = svm_read_problem('dataset\ijcnn1\ijcnn1.t')

model = svm_train(*(train_data))

predict = svm_predict(*(test_data), model)

print(len(predict))
print(predict[1])

准确率：92.78742870852008
~~~





