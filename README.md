# kaggle_ensemble_stacking-for-Titanic

像kaggle、天池这些的数据挖掘大赛将会是我们实践的一个很好的平台。

而泰坦尼克号数据集，则是kaggle上机器学习入门的一个非常好的数据集。树根借鉴网络上一些大神的方案，尝试对泰坦尼克号数据集进行数据挖掘预测，仅以抛砖引玉之用。

模型思路为：

1.第一步我们先要对数据进行可视化，进行简单的数据探索；

2.然后对数据进行清洗，比如对缺失值进行填补、进行特征二值化，编制哑变量等等。值得一提的是，这里由于年龄缺失值较多，树根对年龄的缺失值采用随机森林模型预测填补的方法；

3.然后就是进行我们的特征工程，这一步是最重要的，再用xgboost进行特征选择；

4.最后就是构建我们的模型，在这一篇文章，树根会用到kaggle比赛最常见的构建模型的方法——模型融合（Ensemble）中的stack，base models分别选择随机森林、Extratrees、梯度提升树，最后构建xgboost模型来预测生还率；

5.进行参数调整把模型性能提升到最优（这里树根因为时间关系使用了网格搜索进行调参）。

代码较多，运行顺序为：

```python
import.py
data_visualization.py
data_preprocessing.py
missing_value_processing.py
data_preprocessing_2.py
feature_selection.py
ensemble_stacking.py
evaluate.py
```

使用混淆矩阵，得到最后评估效果：

![image](https://github.com/chenyeroot/kaggle_ensemble_stacking-for-Titanic/blob/master/picture/picture/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5%E7%BB%93%E6%9E%9C.jpg)
