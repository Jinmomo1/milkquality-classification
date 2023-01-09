# milkquality-classification
## 项目简介
本项目根据牛奶相关数据集，包括8个独立变量，即pH值、温度、味道、气味、脂肪、浊度、颜色以及牛奶的品质，建立统计和预测模型来预测牛奶的质量。分别运用了逻辑回归算法和随机森林算法对牛奶的品质进行分类，比较两种模型的拟合效果和预测效果。
（数据来源：https://www.kaggle.com/datasets/cpluzshrijayan/milkquality）

## 复现说明
本项目使用python3进行编程。首先导入以下的模块：
```python
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
```
建立数据预处理的类dataProcessing，类中包含数据集，判断是否存在缺失值的函数以及标准化处理的函数。其中判断是否存在缺失值的函数用到了异常处理，利用try...except...语句判断整个数据集中是否有缺失值，如果没有，输出“数据没有缺失值。”如果存在缺失值，引起异常“Missingdata”，并打印“请先处理缺失值！”
```python
#定义数据预处理的类
class dataProcessing:
    def __init__(self,data):
        self.data=data
    def Missing(self):
        try:
            if (data.isnull().sum().sum()==0):
                print("数据没有缺失值。")
            else:
                raise Exception('Missingdata')
        except:
            print("请先处理缺失值！")
    def Standard(self,i):
        x = data.iloc[:,i]
        meanx = np.mean(x)
        stdx = np.std(x)
        y = (x-meanx)/stdx
        return y
 ```
 将牛奶数据集导入python，并创建类dataProcessing的对象dP，调用函数dP.Missing()，发现结果为“数据没有缺失值。”说明该数据集不存在缺失值。
```python
#路径为电脑中数据所在路径
data = pd.read_csv("D:/用户目录/Administrator/Desktop/编程基础/final/milk/milknew.csv")
data.head()
dP = dataProcessing(data)
dP.Missing()
```
对数据整体进行描述，
```python
data.describe()
```
### 数据可视化：
```python
#对牛奶品质做可视化
plt.figure(figsize=(8,8))
palette_color = sns.color_palette('pastel')
explode = [0.1, 0.1, 0.1]
data.groupby('Grade')['Grade'].count().plot.pie(colors=palette_color,explode=explode, autopct="%1.1f%%");
```
```python
#连续型变量密度图
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.set_style('darkgrid' ) 
for var, subplot in zip(['pH', 'Temprature', 'Colour'], ax.flatten()):
    sns.kdeplot(x= var, data=data, ax=subplot)
```
```python
#观察不同ph、温度和颜色对牛奶品质的影响
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
var = ['pH','Temprature','Colour']
for i in range(3):
    sns.boxplot(x='Grade', y=var[i], data=data, ax=list(ax.flatten())[i], palette=sns.color_palette("pastel", 3))
```
```python
#二分类变量柱状图
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for var, subplot in zip(['Taste', 'Odor', 'Fat ', 'Turbidity'], ax.flatten()):
    sns.histplot(data=data[var],ax=subplot)
```
```python
#二分类变量和牛奶品质
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for var, subplot in zip(['Taste', 'Odor', 'Fat ', 'Turbidity'], ax.flatten()):
    sns.barplot(x='Grade', y= var, data=data, ax=subplot, palette=sns.color_palette("pastel", 3))
```
```python
#相关性热图
f, axes = plt.subplots(1, 1, figsize=(6, 5))
sns.heatmap(data.corr(), vmin = -1, vmax = 1,  linewidths = 1,
           annot = True, fmt = ".2f", annot_kws = {"size": 14}, cmap = "YlGnBu")
```
### 建模：

将牛奶品质Grade数据列的low, medium, high分别替换为0，1，2. 由于pH，温度和颜色的数值大小相差较大，先对它们进行标准化处理。对这三列数据利用dP调用之前类的标准化函数，得到新的数据集data2. 
```python
#将牛奶品质相应替换为数字0，1，2
data=data.replace({'low': 0, 'medium': 1, 'high': 2,})
data.head()
```
```python
#标准化数据
ph2 = dP.Standard(0)
Temprature2 = dP.Standard(1)
Colour2 = dP.Standard(6)
data2 = data.copy()
data2['pH'] = ph2
data2['Temprature'] = Temprature2
data2['Colour'] = Colour2
data2.head()
```
导入以下模块：
```python
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.metrics import confusion_matrix #for confusion matrix 
from sklearn.metrics import plot_confusion_matrix
```
将data2数据集的Grade列分离出来为y，剩下7个变量组成数据集x，再利用train_test_split函数将数据按照7：3划分为训练集和测试集。
```python
#分离训练集和测试集
x= data2.drop(['Grade'],axis=1)
y= data2['Grade']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=30)
print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)
```

#### 逻辑回归
利用网格搜索法对逻辑回归模型进行调参。选取C：正则强度的倒数以及solver：优化算法选择参数这两个参数进行搜索。
```python
#from sklearn.model_selection import GridSearchCV  # 网格搜索
p = {
    'C':list(np.linspace(0.05,1,19)),
    'solver':['liblinear','sag','newton-cg','lbfgs']}

model = LogisticRegression()
GS = GridSearchCV(model,p,cv=5,scoring='accuracy')
GS.fit(X_train,y_train)
#输出最优参数
GS.best_params_
```
选取最佳参数在训练集上进行模型拟合，得到模型准确度为 85.29 %.代码如下：
```python
#选择最佳参数进行模型拟合
LR_model = LogisticRegression(C=0.84,solver='sag')
LR_model.fit(X_train,y_train)
#训练集拟合效果
train_accuracy = round(LR_model.score(X_train, y_train)*100,2)
print("Training Accuracy: % {}".format(train_accuracy))
```
绘制混淆矩阵
```python
plot_confusion_matrix(LR_model, X_train, y_train )  
plt.show()
```
在测试集上预测，结果准确度为85.53 %. 
```python
#测试集预测效果
test_accuracy = round(LR_model.score(X_test, y_test)*100,2)
print("Testing Accuracy: % {}".format(test_accuracy))
```
绘制混淆矩阵
```python
plot_confusion_matrix(LR_model, X_test, y_test )  
plt.show()
```
#### 随机森林
利用网格搜索法对随机森林模型进行调参。选取随机森林的子评估器数量，树的深度这两个参数进行搜索，得出最优参数。
```python
param_grid={
    'n_estimators':range(20,120,10),
    'max_depth':range(5,13), 
}
 
grid = GridSearchCV(RandomForestClassifier(),
          param_grid,cv=5,scoring='accuracy')
 
grid.fit(X_train,y_train)
print(grid.best_params_)
```
选取最佳参数在训练集上进行模型拟合，得到模型准确度为 100 %.代码如下：
```python
#选择最佳参数进行模型拟合
RF_model = RandomForestClassifier(max_depth=7,n_estimators=30)
RF_model.fit(X_train,y_train)
#训练集拟合效果
train_accuracy = round(RF_model.score(X_train, y_train)*100,2)
print("Training Accuracy: % {}".format(train_accuracy))
```
绘制混淆矩阵
```python
plot_confusion_matrix(RF_model, X_train, y_train )  
plt.show()
```
在测试集上预测，结果准确度为100 %. 
```python
#测试集预测效果
#测试集预测效果
test_accuracy = round(RF_model.score(X_test, y_test)*100,2)
print("Testing Accuracy: % {}".format(test_accuracy))
```
绘制混淆矩阵
```python
plot_confusion_matrix(RF_model, X_test, y_test )  
plt.show()
```
