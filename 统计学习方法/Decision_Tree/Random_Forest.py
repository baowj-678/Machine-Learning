""" 随机森林
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/9/22
"""
import numpy as np
import math
import copy
from Decision_Tree import DecisionTree

class RandomForest():
    def __init__(self):
        super().__init__()
        