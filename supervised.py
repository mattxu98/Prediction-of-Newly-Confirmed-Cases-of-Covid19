# 1. 数据预处理
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('湖北.xlsx')
df



#均值
newcase_mean = df['新增确诊病例'].mean()
print(newcase_mean)
#标准差
newcase_std = df['新增确诊病例'].std()
print(newcase_std)
#在mean±3*std之外为异常值
newcase_min = newcase_mean - 3 * newcase_std
newcase_max = newcase_mean + 3 * newcase_std
df[df['新增确诊病例'].map(lambda x:(x > newcase_max) or (x < newcase_min))]



df['新增确诊病例'].iloc[43] = (df['新增确诊病例'].iloc[42] + df['新增确诊病例'].iloc[44])/2
df['新增确诊病例'].iloc[43]



df['下日新增确诊'] = list(df['新增确诊病例'][1:])+[np.nan]
df



df.columns



x = np.array(df[['新增确诊病例', '新增治愈出院数', '新增死亡数', '核减', '治愈核减', '死亡核减','扣除核减累计确诊人数', '累计治愈人数',
        '累计死亡人数', '累计确诊人数']][:-1])
#切去df的第一列：公开时间；切去df的最后一列：下日新增确诊(y)
#最后一行的y是NaN,切去



#对x做标准化处理
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x



#对y做natural log处理
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
y = np.array(df['下日新增确诊'][:-1])#最后一行的y是NaN,切去
y = np.reshape(y,(-1,1))
y = FunctionTransformer(log1p).fit_transform(y)
y



# 2. 特征工程
#将预处理过的x转化为dataframe，方便建立新特征
x = pd.DataFrame(x,index = df['公开时间'].iloc[:-1],columns = ['新增确诊病例', '新增治愈出院数', '新增死亡数', '核减', '治愈核减', 
        '死亡核减','扣除核减累计确诊人数', '累计治愈人数','累计死亡人数', '累计确诊人数'])
x



x['现存病例'] = x['新增确诊病例'] - x['新增治愈出院数'] - x['新增死亡数']
x['净出院数'] = x['新增治愈出院数'] - x['新增死亡数']
x['净累计治愈数'] = x['累计治愈人数'] - x['累计确诊人数']
x['首例天数'] = list(range(len(x))) 
x




# 特征选择
# (1) 此ipynb文件最后结果是不使用PCA的，那么各种模型.fit(x_train,y_train)，不跑此“特征选择”部分代码
# (2) 此部分是使用PCA去除共线性，那么不设置n_components，各种模型.fit(x_train_pca,y_train)
# (3) 如果使用PCA降维,那么n_components需要设置，各种模型.fit(x_train_pca,y_train)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.decomposition import PCA

#实例化PCA
pca = PCA()
#训练模型
pca.fit(x_train)



#对训练集和测试集进行降维
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
x_train_pca



x_test_pca



#各自的方差贡献率
ratio = pca.explained_variance_ratio_
#累计贡献率
cum_ratio = ratio.cumsum()
cum_ratio



import matplotlib.pyplot as plt
#画图展示
xs = list(range(len(cum_ratio)))
plt.plot(xs,cum_ratio)
plt.xlabel('number of components')
plt.ylabel('cum_explained_variance_ratio')
plt.grid()
plt.show()



# 3. 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)



# 4. 选择模型

# 线性回归
from sklearn.linear_model import LinearRegression
#实例化线性回归
clf_lr = LinearRegression()
#对训练集的x和y进行训练
clf_lr.fit(x_train,y_train)



# LinearRegression的参数基本上都是bool，无需网格调参
y_train_pred_lr = np.exp(clf_lr.predict(x_train))-1
y_test_pred_lr = np.exp(clf_lr.predict(x_test))-1



#MSE
from sklearn.metrics import mean_squared_error
MSE_train_lr = mean_squared_error(np.exp(y_train)-1,y_train_pred_lr)
MSE_test_lr = mean_squared_error(np.exp(y_test)-1,y_test_pred_lr)
print('The MSE_train is {:.2f}.\nThe MSE_test is {:.2f}.'.format(MSE_train_lr,MSE_test_lr))



#r^2
from sklearn.metrics import r2_score
r2_train_lr = r2_score(np.exp(y_train)-1,y_train_pred_lr)
r2_test_lr = r2_score(np.exp(y_test)-1,y_test_pred_lr)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_lr,r2_test_lr))



import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 4))
#分别可视化训练集和测试集的预测效果

plt.subplot(1,2,1)
x = list(range(len(y_train)))#设置一个x
plt.plot(x,np.exp(y_train),label = 'true')#实际的y
plt.plot(x,y_train_pred_lr,label = 'pred')#预测的y
plt.legend()
plt.title('Train Set')

plt.subplot(1,2,2)
x = list(range(len(y_test)))
plt.plot(x,np.exp(y_test),label = 'true')
plt.plot(x,y_test_pred_lr,label = 'pred')
plt.legend()
plt.title('Test Set')

plt.show()



# Lasso回归

from sklearn.linear_model import Lasso
#实例化lasso回归
clf_la = Lasso(alpha = 0.001)
clf_la.fit(x_train,y_train)



y_train_pred_la = np.exp(clf_la.predict(x_train))-1
y_test_pred_la = np.exp(clf_la.predict(x_test))-1



#MSE
MSE_train_la = mean_squared_error(np.exp(y_train)-1,y_train_pred_la)
MSE_test_la = mean_squared_error(np.exp(y_test)-1,y_test_pred_la)
print('The MSE_train is {:.2f}.\nThe MSE_test is {:.2f}.'.format(MSE_train_la,MSE_test_la))



#r^2
r2_train_la = r2_score(np.exp(y_train)-1,y_train_pred_la)
r2_test_la = r2_score(np.exp(y_test)-1,y_test_pred_la)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_la,r2_test_la))



fig = plt.figure(figsize=(12, 4))
#分别可视化训练集和测试集的预测效果

plt.subplot(1,2,1)
x = list(range(len(y_train)))#设置一个x
plt.plot(x,np.exp(y_train),label = 'true')#实际的y
plt.plot(x,y_train_pred_la,label = 'pred')#预测的y
plt.legend()
plt.title('Train Set')

plt.subplot(1,2,2)
x = list(range(len(y_test)))
plt.plot(x,np.exp(y_test),label = 'true')
plt.plot(x,y_test_pred_la,label = 'pred')
plt.legend()
plt.title('Test Set')

plt.show()



# 决策树回归

# 无网格搜索调参（目的：查看参数原始状态；与调参后模型对比拟合效果）
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 3)#max_depth必须有，否则MSE_train = 0.0
dtr.fit(x_train,y_train)
# max_features,max_leaf_nodes比min_samples_leaf,min_samples_split更值得调



y_train_pred_dtr = np.exp(dtr.predict(x_train))-1
y_test_pred_dtr = np.exp(dtr.predict(x_test))-1



r2_train_dtr = r2_score(np.exp(y_train)-1,y_train_pred_dtr)
r2_test_dtr = r2_score(np.exp(y_test)-1,y_test_pred_dtr)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_dtr,r2_test_dtr))



# 网格搜索调参
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time
start = time.time()
#实例化模型
dtr = DecisionTreeRegressor(random_state = 0)
#参数
params = {'max_depth':list(range(3,10,1)),'max_features':list(range(5,16,2)),'max_leaf_nodes':list(range(5,20,2))}
#实例化网格搜索
gs_dtr = GridSearchCV(estimator = dtr,param_grid = params,cv = 5,scoring = make_scorer(r2_score))
#训练数据
gs_dtr.fit(x_train,y_train)
end = time.time()
#查看最优参数
print(gs_dtr.best_params_)
#查看最优参数下的指标情况
print(gs_dtr.best_score_)
print('网格搜索所消耗的时间为：%.3f S'% float(end - start))

# 将make_scorer(r2_score)置换为make_scorer(mean_squared_error,greater_is_better=False)，调参结果相同



# 结果
dtr = DecisionTreeRegressor(max_depth = 5,max_features = 11,max_leaf_nodes = 15,random_state = 1)
dtr.fit(x_train,y_train)



y_train_pred_dtr = np.exp(dtr.predict(x_train))-1
y_test_pred_dtr = np.exp(dtr.predict(x_test))-1



MSE_train_dtr = mean_squared_error(np.exp(y_train)-1,y_train_pred_dtr)
MSE_test_dtr = mean_squared_error(np.exp(y_test)-1,y_test_pred_dtr)
print('The MSE_train is {:.2f}.\nThe MSE_test is {:.2f}.'.format(MSE_train_dtr,MSE_test_dtr))



r2_train_dtr = r2_score(np.exp(y_train)-1,y_train_pred_dtr)
r2_test_dtr = r2_score(np.exp(y_test)-1,y_test_pred_dtr)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_dtr,r2_test_dtr))



fig = plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
x = list(range(len(y_train)))
plt.plot(x,np.exp(y_train),label = 'true')
plt.plot(x,y_train_pred_dtr,label = 'pred')
plt.legend()
plt.title('Train Set')

plt.subplot(1,2,2)
x = list(range(len(y_test)))
plt.plot(x,np.exp(y_test),label = 'true')
plt.plot(x,y_test_pred_dtr,label = 'pred')
plt.legend()
plt.title('Test Set')

plt.show()



# 随机森林回归
# 无网格搜索调参（目的：查看参数原始状态；与调参后模型对比拟合效果）
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)



y_train_pred_rfr = np.exp(rfr.predict(x_train))-1
y_test_pred_rfr = np.exp(rfr.predict(x_test))-1



MSE_train_rfr = mean_squared_error(np.exp(y_train)-1,y_train_pred_rfr)
MSE_test_rfr = mean_squared_error(np.exp(y_test)-1,y_test_pred_rfr)
print('The MSE_train is {:.2f}.\nThe MSE_test is {:.2f}.'.format(MSE_train_rfr,MSE_test_rfr))



r2_train_rfr = r2_score(np.exp(y_train)-1,y_train_pred_rfr)
r2_test_rfr = r2_score(np.exp(y_test)-1,y_test_pred_rfr)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_rfr,r2_test_rfr))



# 网格搜索调参
start = time.time()
#实例化模型
rfr = RandomForestRegressor(random_state = 1)
#参数
params = {'max_depth':list(range(3,10,1)),'max_features':list(range(5,16,2)),'max_leaf_nodes':list(range(5,16,2)),\
          'n_estimators':list(range(5,16,2))}
#实例化网格搜索
gs_rfr = GridSearchCV(estimator = rfr,param_grid = params,cv = 5,scoring = make_scorer(r2_score))
#训练数据
gs_rfr.fit(x_train,y_train)
end = time.time()
#查看下最优参数
print(gs_rfr.best_params_)
#查看下最优参数下的指标情况
print(gs_rfr.best_score_)
print('网格搜索所消耗的时间为：%.3f S'% float(end - start))

# 将make_scorer(r2_score)置换为make_scorer(mean_squared_error,greater_is_better=False)，调参结果相同



# 结果
rfr = RandomForestRegressor(max_depth = 3,max_features = 13,max_leaf_nodes = 7,n_estimators = 5,random_state = 2)
rfr.fit(x_train,y_train)



y_train_pred_rfr = np.exp(rfr.predict(x_train))-1
y_test_pred_rfr = np.exp(rfr.predict(x_test))-1



MSE_train_rfr = mean_squared_error(np.exp(y_train)-1,y_train_pred_rfr)
MSE_test_rfr = mean_squared_error(np.exp(y_test)-1,y_test_pred_rfr)
print('The MSE_train is {:.2f}.\nThe MSE_test is {:.2f}.'.format(MSE_train_rfr,MSE_test_rfr))



r2_train_rfr = r2_score(np.exp(y_train)-1,y_train_pred_rfr)
r2_test_rfr = r2_score(np.exp(y_test)-1,y_test_pred_rfr)
print('The r2_train is {:.4f}.\nThe r2_test is {:.4f}.'.format(r2_train_rfr,r2_test_rfr))



fig = plt.figure(figsize=(12, 4))

plt.subplot(1,2,1)
x = list(range(len(y_train)))
plt.plot(x,np.exp(y_train),label = 'true')
plt.plot(x,y_train_pred_rfr,label = 'pred')
plt.legend()
plt.title('Train Set')

plt.subplot(1,2,2)
x = list(range(len(y_test)))
plt.plot(x,np.exp(y_test),label = 'true')
plt.plot(x,y_test_pred_rfr,label = 'pred')
plt.legend()
plt.title('Test Set')

plt.show()



# 5. 对比MSE, r2
#这是没有用PCA的结果
columns_ = ['MSE_train','MSE_test','r2_train','r2_test']
index_ = ['lr','la','dtr','rfr']
res = pd.DataFrame(index = index_,columns = columns_)
for i in columns_:
    for j in index_:
        name = i + '_' + j
        res[i][j] = locals()[name]
res.index = ['Linear Regression','Lasso Regression','Decision Tree Regression','Random Forest Regression']
res

# 结论1：在不使用PCA时，决策树回归是最佳模型，因为其MSE最小而r2最大

res2 = pd.DataFrame([[170623,744323,0.857063,0.329477],[254754,372709,0.786583,0.664245],[174575,321313,0.853753,0.710545],\
                   [85292.8,87643.5,0.928547,0.921046]],index = res.index,columns = res.columns)
res2

# 结论2：在使用PCA去除共线性时，随机森林回归是最佳模型，因为其MSE最小而r2最大