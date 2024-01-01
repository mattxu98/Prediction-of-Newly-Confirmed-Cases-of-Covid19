import numpy as np
import pandas as pd
df = pd.read_excel('湖北.xlsx')
df.head()
df.info()
df.columns



x = df[['公开时间', '新增确诊病例', '新增治愈出院数', '新增死亡数', '核减', '治愈核减', '死亡核减','扣除核减累计确诊人数', '累计治愈人数',
        '累计死亡人数', '累计确诊人数', '现存病例', '净出院数', '净累计治愈数','首例天数']][:-1]
y = df['下日新增确诊'][:-1]#最后一行的y是NaN,切去
x
y



# pd.dataframe转换成np.array才能嵌入机器学习模型
x = np.array(x)[:,1:]#切去df的第一列：时间标签
x



y = np.array(y)
print(x.shape)
print(y.shape)
y



# 线性回归
#使用sklearn线性回归的库
from sklearn.linear_model import LinearRegression

#实例化一个线性回归
clf = LinearRegression()
#需要根据已知的x和y进行训练
clf.fit(x,y)



#查看每个x对应的系数
print(clf.coef_)
#查看截距项
print(clf.intercept_)



y_pred = clf.predict(x)
y_pred



from sklearn.metrics import mean_squared_error

mean_squared_error(y,y_pred)



#画图展示
import matplotlib.pyplot as plt

#设置一个x
x1 = list(range(len(y)))
plt.plot(x1,y,label = 'true')#实际的y
plt.plot(x1,y_pred,label = 'pred')#预测的y
plt.legend()#展示下图例
plt.show()



# 交叉验证
#划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)



#调用线性模型
from sklearn.linear_model import LinearRegression

#使用的是训练集上的X和Y
clf = LinearRegression()
clf.fit(x_train,y_train)



clf.coef_



y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)



#MSE
from sklearn.metrics import mean_squared_error
#训练集
mean_squared_error(y_train,y_train_pred)
#测试集
mean_squared_error(y_test,y_test_pred)



#R2
from sklearn.metrics import r2_score
#训练集
r2_score(y_train,y_train_pred)
#测试集
r2_score(y_test,y_test_pred)



#看一下我们的MAE
from sklearn.metrics import mean_absolute_error
#训练集
mean_absolute_error(y_train,y_train_pred)
#测试集
mean_absolute_error(y_test,y_test_pred)



# 分类_决策树
## 数据集划分
# from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.2,random_state=0)



#导入决策树模型（分类）
from sklearn.tree import DecisionTreeClassifier

#实例化模型
model = DecisionTreeClassifier(max_depth = 3)



#训练模型
model.fit(x_train1,y_train1)



#预测数据
y_train_pred1 = model.predict(x_train1)
y_test_pred1 = model.predict(x_test1)



#准确率
from sklearn.metrics import accuracy_score
accuracy_score(y_train_pred1,y_train1)



# from sklearn.metrics import mean_squared_error
mean_squared_error(y_train_pred1,y_train1)
mean_squared_error(y_test_pred1,y_test1)
# from sklearn.metrics import r2_score
r2_score(y_train_pred1,y_train1)
r2_score(y_test_pred1,y_test1)



# 随机森林
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error



x_train2,x_test2,y_train2,y_test2 = train_test_split(x,y,test_size = 0.2)



rfr = RandomForestRegressor()
rfr.fit(x_train2,y_train2)



y_train_pred2 = rfr.predict(x_train2)
y_test_pred2 = rfr.predict(x_test2)



mean_squared_error(y_train_pred2,y_train2)



mean_squared_error(y_test_pred2,y_test2)



#调节参数，决策树的棵树
error_trains = []
error_tests = []
for i in range(10,200,10):
    #修改n_estimators，和确定随机种子
    rfr = RandomForestRegressor(n_estimators = i, random_state = 90)# n_estimators是决策树的数量
    rfr.fit(x_train2,y_train2)
    train_predicts = rfr.predict(x_train2)
    test_predicts = rfr.predict(x_test2)
    error_trains.append(mean_squared_error(y_train_pred2,y_train2))
    error_tests.append(mean_squared_error(y_test_pred2,y_test2))



import matplotlib.pyplot as plt
#绘制调参图
x = list(range(10,200,10))
plt.plot(x,error_trains,'--',label = 'train')
plt.plot(x,error_tests,'o-',label = 'test')
plt.ylabel('MSE')
plt.xlabel('n_estimators')
plt.legend()
plt.show()



# 调参
# 网格搜索
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")



#开始时间
start = time.time()
#实例化模型
lr = RandomForestClassifier(random_state = 0)
#参数
params = {'max_depth':list(range(3,10,1)),'n_estimators':list(range(10,31,2))}
#实例化网格搜索
gs_lr = GridSearchCV(estimator = lr,param_grid = params,cv = 5,scoring = make_scorer(accuracy_score))
#训练数据
gs_lr.fit(x_train,y_train)
#结束时间
end = time.time()
#查看最优参数
print(gs_lr.best_params_)
#查看最优参数下的指标情况
print(gs_lr.best_score_)
print('网格搜索所消耗的时间为：%.3f S'% float(end - start))



#测试集的准确率
gs_lr.score(x_test,y_test)



#查看调参的过程
gs_lr.cv_results_



# 随机网格搜索RandomizedSearchCV()
#开始时间
start = time.time()
#实例化模型
lr = RandomForestClassifier(random_state = 0)
#参数
params = {'max_depth':list(range(3,10,1)),'n_estimators':list(range(10,31,2))}
#实例化网格搜索
gs_lr = RandomizedSearchCV(estimator = lr,param_distributions = params,cv = 5,scoring = make_scorer(accuracy_score))
#训练数据
gs_lr.fit(x_train,y_train)
#结束时间
end = time.time()
#查看最优参数
print(gs_lr.best_params_)
#查看最优参数下的指标情况
print(gs_lr.best_score_)
print('随机网格搜索所消耗的时间为：%.3f S'% float(end - start))



#查看测试集的准确率
gs_lr.score(x_test,y_test)


#查看调参的过程
gs_lr.cv_results_



# 在notebook运行 conda install -c conda-forge scikit-learn=0.22.2 否则TypeError: init() got an unexpected keyword argument 'iid'
# source:https://stackoverflow.com/questions/63597330/downgrade-sklearn-0-22-3-to-0-22-2-in-anaconda-prompt



# 贝叶斯搜索BayesSearchCV()
#开始时间
start = time.time()
#实例化模型
lr = RandomForestClassifier(random_state = 0)
#参数
params = {'n_estimators':list(range(10,31,2)),'max_depth':list(range(3,10,1))}
#实例化网格搜索
gs_lr = BayesSearchCV(lr,params,cv = 5,scoring = make_scorer(accuracy_score),n_iter = 40)
#训练数据
gs_lr.fit(x_train,y_train)
#结束时间
end = time.time()
#查看下最优参数
print(gs_lr.best_params_)
#查看下最优参数下的指标情况
print(gs_lr.best_score_)
print('贝叶斯搜索所消耗的时间为：%.3f S'% float(end - start))



#查看测试集的准确率
gs_lr.score(x_test,y_test)



#查看调参的过程
gs_lr.cv_results_



# 特征工程 / 数据预处理
df = pd.read_excel('湖北.xlsx')
df.columns

df

x = df[['新增确诊病例', '新增治愈出院数', '新增死亡数', '核减', '治愈核减', '死亡核减','扣除核减累计确诊人数', '累计治愈人数',
        '累计死亡人数', '累计确诊人数', '现存病例', '净出院数', '净累计治愈数','首例天数']][:-1] 
                                                                                #把'公开时间'切去不用，把'下日新增确诊'切去作为y
y = df['下日新增确诊'][:-1]#最后一行的y是NaN,切去
x

y

#标准化
xstd = (x - x.mean()) / x.std()
xstd


# 无监督学习
# 聚类
# 手肘法
from sklearn.cluster import KMeans
result_list = []#存储所有聚类中心均值向量的总和
#假设类别是2-11
for i in range(2,12):
    #新建聚类模型，改变聚类中心个数
    model = KMeans(n_clusters = i, random_state = 1)
    #训练模型
    model.fit(xstd)
    #取出所有聚类中心均值向量的总和
    result_list.append(model.inertia_)



#手肘图
import matplotlib.pyplot as plt

xs = list(range(2,12))
plt.plot(xs, result_list)
plt.show()



# 轮廓系数
from sklearn import metrics
result_list1 = []
#假设类别是2-11
for i in range(2,12):
    #新建聚类模型，改变聚类中心个数
    model = KMeans(n_clusters = i,random_state = 1)
    #训练模型
    model.fit(xstd)
    #获取KMeans聚类的结果
    labels = model.labels_
    #计算轮廓系数
    res = metrics.silhouette_score(xstd,labels)
    result_list1.append(res)



xs = list(range(2,12))
plt.plot(xs, result_list1)
plt.show()



# 根据手肘法和轮廓系数的图像，k=4
#新建聚类模型，改变聚类中心个数
model = KMeans(n_clusters = 4,random_state = 1)
#训练模型
model.fit(xstd)
#获取kmeans聚类的结果
labels = model.labels_
print(len(labels))
labels



# 对y，'下日新增确诊'，等频离散化
df1 = df[['新增确诊病例', '新增治愈出院数', '新增死亡数', '核减', '治愈核减', '死亡核减','扣除核减累计确诊人数', '累计治愈人数',
        '累计死亡人数', '累计确诊人数', '现存病例', '净出院数', '净累计治愈数','首例天数','下日新增确诊']][:-1]
df1['labels'] = labels
#等频离散化
df1['等频划分下日新增'] = pd.qcut(df1['下日新增确诊'],q = 4)
df1



#将等频划分数据映射为0,1,2,3
df1['等频划分下日新增'].value_counts()



df1['等频划分下日新增']



#转化为字符串类型
df1['等频划分下日新增'] = df1['等频划分下日新增'].map(lambda x:str(x))
#根据字典进行映射
dis = {'(-0.001, 41.0]':0,
       '(41.0, 366.0]':1,
       '(366.0, 1692.0]':2,
       '(1692.0, 14840.0]':3}
df1['等频划分下日新增'] = df1['等频划分下日新增'].map(dis)



df1



# 展示聚类结果
list(set(df1['等频划分下日新增']))



#真实的结果散点图
for label in list(set(df1['等频划分下日新增'])):
    df2 = df1[df1['等频划分下日新增'] == label]
    plt.plot(df2['首例天数'],df2['下日新增确诊'],'o',label = label)
plt.legend()
plt.show()



#聚类的结果散点图
for label in list(set(df1['labels'])):
    df2 = df1[df1['labels'] == label]
    plt.plot(df2['首例天数'],df2['下日新增确诊'],'o',label = label)
plt.legend()
plt.show()



# 降维
from sklearn.decomposition import PCA
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error



#实例化PCA
pca = PCA()
#训练模型
pca.fit(x_train,y_train)



#返回各自的方差贡献率
ratio = pca.explained_variance_ratio_
ratio



#计算累计贡献率
cum_ratio = ratio.cumsum()
cum_ratio



#画图展示
xs = list(range(len(cum_ratio)))
plt.plot(xs,cum_ratio)
plt.xlabel('number of components')
plt.ylabel('cum_explained_variance_ratio')
plt.grid()
plt.show()