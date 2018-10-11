# 导入相应的工具库
# 数据可视化部分会用到Pyecharts，请自行安装
import pandas as pd  
from pyecharts import Bar
import matplotlib.pylab as plt

# 导入数据集并进行可视化
train = pd.read_excel('.../train.xlsx')
test = pd.read_excel('.../test.xlsx')
y_predict = pd.read_excel('.../y_test.xlsx')
y_predict = y_predict.iloc[:,1]

train.head(10)

test.head(10)

# 接下来查看数据的缺失情况，isnull（）函数返回缺失值的布尔值，然后使用sum（）把缺失值加总
print(train.isnull().sum())
print("------我是分割线-----")
# 题外话，传入how='all'则之丢弃全为NaN的行或列
#data = data_1.dropna(how='all',axis=1)
#data = data_1.dropna(thresh=5000,axis=1)
print(test.isnull().sum())

# 把生还和死亡的人数统计起来
Pclass_Survived_0 = train.Pclass[train.Survived == 0].value_counts() 
Pclass_Survived_1 = train.Pclass[train.Survived == 1].value_counts()
Pclass_Survived = pd.DataFrame({'生还':Pclass_Survived_1, '死亡':Pclass_Survived_0}) 

# 可以自行搜一下RGB颜色对照表
color_0 = '#9AFF9A'
color_1 = '#8B8B7A'

# 各舱位等级的获救情况的可视化
attr = ["等级1","等级2","等级3"]
v1 = Pclass_Survived["死亡"]
v2 = Pclass_Survived["生还"]
bar_1 = Bar("各舱位等级的获救情况")
bar_1.add("死亡人数", attr, v1, is_label_show=True,label_color= [color_0])
bar_1.add("生还人数", attr, v2, is_label_show=True,label_color= [color_1])
bar_1

# 各登录港口乘客的获救情况的可视化
Embarked_Survived_0 = train.Embarked[train.Survived == 0].value_counts() 
Embarked_Survived_1 = train.Embarked[train.Survived == 1].value_counts()
Embarked_Survived = pd.DataFrame({'生还':Embarked_Survived_1, '死亡':Embarked_Survived_0}) 

attr = ["港口S","港口C","港口Q"]
v1 = Embarked_Survived["死亡"]
v2 = Embarked_Survived["生还"]
bar_2 = Bar("各登录港口乘客的获救情况")
bar_2.add("死亡人数", attr, v1, is_label_show=True,is_stack=True,label_color= [color_0])
bar_2.add("生还人数", attr, v2, is_label_show=True,is_stack=True,label_color=[color_1])
bar_2

# 各性别乘客的获救情况
Sex_Survived_0 = train.Sex[train.Survived == 0].value_counts() 
Sex_Survived_1 = train.Sex[train.Survived == 1].value_counts()
Sex_Survived = pd.DataFrame({'生还':Sex_Survived_1, '死亡':Sex_Survived_0}) 

attr = ["女性","男性"]
v1 = Sex_Survived["死亡"]
v2 = Sex_Survived["生还"]
bar_3 = Bar("各性别乘客的获救情况")
bar_3.add("死亡人数", attr, v1, is_stack=True,label_color= [color_0])
bar_3.add("生还人数", attr, v2, is_label_show=True,is_stack=True,label_color= [color_1])
bar_3
