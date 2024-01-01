# 导入依赖库
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm  # 制作进度条
import matplotlib.pyplot as plt
%matplotlib inline

df1 = pd.read_excel('湖北.xlsx',index_col = 0)
#查看数据的特征
print('数据集的特征：{}\n'.format(df1.columns.values))
#查看数据的index
print('数据集的Index：{}\n'.format(df1.index))
#查看数据的shape
print('数据集的形状：{}'.format(df1.shape))



# TODO：画出特征分布的直方图
fig = plt.figure(figsize=(15,15))
for i in range(len(df1.columns)): 
    axe = fig.add_subplot(4,3,i+1)   
    axe.hist(df1.iloc[:,i],bins=20)
    plt.title(df1.columns[i])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False



# TODO：画出特征分布的折线图
fig = plt.figure(figsize=(15,15))
for i in range(len(df1.columns)): 
    axe = fig.add_subplot(4,3,i+1)   
    axe.plot(df1.iloc[:,i],color='r')
    plt.title(df1.columns[i])
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False



# TODO：统计数据集中有多少行/列有缺失
print('数据集中有',(df1.isnull().sum(axis=1)>0).sum(),'行缺失')
print('数据集中有',(df1.isnull().sum(axis=0)>0).sum(),'列缺失')



df2 = df1['新增确诊病例']
df2
#均值
newcase_mean = df2.mean()
print(newcase_mean)
#标准差
newcase_std = df2.std()
print(newcase_std)
#在mean±3*std之外为异常值
newcase_min = newcase_mean - 3 * newcase_std
newcase_max = newcase_mean + 3 * newcase_std
df2[df2.map(lambda x:(x > newcase_max) or (x < newcase_min))]
df2.iloc[43] = (df2.iloc[42] + df2.iloc[44])/2
df2.iloc[43]



# ADF检验：该p值无法拒绝原假设：序列为非平稳序列。所以该序列为非平稳序列
# 白噪声检验：该p值拒绝原假设：序列为白噪声。所以该序列非白噪声

from statsmodels.tsa.stattools import adfuller #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验

def adf_box(series):
    '''Return the result of ADF test and Ljung-box test'''
    t = adfuller(series)
    output=pd.DataFrame(index=['ADF Test', 'Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",\
                               "Critical Value(1%)","Critical Value(5%)","Critical Value(10%)", 'Ljung-Box Test','p value'],\
                                columns=['value'])
    output['value']['ADF Test'] = ' '
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    output['value']['Ljung-Box Test'] = ' '
    output['value']['p value'] = acorr_ljungbox(series,lags=1)[1][0]
    return output

adf_box(df2)



# ARIMA模型
from numpy import log1p

#函数1  移动平均+差分
def plot_data(Series_1):
    '''Return the data after moving average and first order differentiation'''
    # 7天为单位进行移动平均
    rol_mean = Series_1.rolling(window=7).mean()
    # 一阶差分
    diff_1 = rol_mean.diff(1)
    # 填充缺失
    rol_mean.fillna(0,inplace=True)
    diff_1.fillna(0,inplace=True)
    return rol_mean,diff_1

plot_data(df2)



# plot_data(df2)[1]是函数1的diff_1
# 不能通过ADF检验，通过Ljung-Box白噪声检验
adf_box(plot_data(df2)[1])



import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm 

#函数2  自动定阶
def arima_order_select_both(Series_1,p_max=4,q_max=4):
    # 搜索AR和MA的参数，以AIC为标准
    res = sm.tsa.arma_order_select_ic(Series_1,ic=['aic'],max_ar=p_max,max_ma=q_max)
    return res.aic_min_order[0],res.aic_min_order[1]

arima_order_select_both(plot_data(df2)[1])



from statsmodels.tsa.arima_model import ARIMA #模型

#函数3  返回预测值 (被函数4反复调用)
def arima_model(Series_1,rol_mean,prim_data,ij,order=(2,1,0)):
    # 初始化模型，待入自动定阶的参数
    model = ARIMA(Series_1,order=order)  
    # 拟合模型
    result_arima = model.fit(disp=-1,transparams=False, method='css')
    # 获得预测值
    predict_ts = result_arima.predict()
    # 一阶差分还原（还原模型中的差分）
    diff_shift_ts = Series_1.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts) 
    # 再次一阶差分还原（还原建模前的差分）
    rol_shift_ts = rol_mean.shift(1)
    diff_recover = diff_recover_1.add(rol_shift_ts)
    # 移动平均还原
    rol_sum = prim_data.rolling(window=6).sum()
    rol_recover = diff_recover * 7 - rol_sum.shift(1)
    # 对数还原
    log_recover = np.exp(rol_recover-1)
    log_recover.dropna(inplace=True)
    # 获得预测值
    r = log_recover
    # 获得初次模型预测值
    if ij==0:
        return r
    # 获得滚动预测值
    else:
        return r.iloc[-1]
    


#函数4  生成ARIMA预测数据集
def arima_model_pred(Series_1,step=12,p_max=4,q_max=4):
    # 空列表储存结果
    arima_pred = []
    for ij in range(len(Series_1)-step+1):
        # 滚动获取数据，并取对数
        if ij+step<len(Series_1):
            dtf = np.log(Series_1[:ij+step-1]+1)
        else:
            dtf = np.log(Series_1[:]+1)
        # 获得移动平均结果和一阶差分结果，套用函数1
        rol_mean,diff_1 = plot_data(dtf)
        # 获得参数，套用函数2
        p = arima_order_select_both(diff_1,p_max,q_max)[0]
        q = arima_order_select_both(diff_1,p_max,q_max)[1]
        # 套用函数3，获得预测值，注意自动定阶可能不稳定，构建模型失败
        # 对失败的模型直接取上一次的预测值作为本次预测
        try:
            a_pred = arima_model(diff_1,rol_mean,dtf,ij,order=(p,1,q))
        except:
            a_pred = a_pred  
        # 获得初次模型预测值
        if ij==0:
            arima_pred.extend(a_pred)
            print('初次运行的12天数据得到',len(a_pred),'天预测')
        # 获得滚动预测值
        else:
            arima_pred.append(a_pred)
            print('第',ij,'次数据进行中')
    return arima_pred



data_arima_p = arima_model_pred(df2,step=12,p_max=4,q_max=4)
print(data_arima_p)
print('预测长度为：',len(data_arima_p))



# 保存data_arima_p
import pickle # 使用pickle保存python对象
data_arima_p_file = 'data_arima_p.data'
f = open(data_arima_p_file, 'wb')
pickle.dump(data_arima_p, f)
f.close()



# 加载data_arima_p
import pickle # 加载对象
data_arima_p_file = 'data_arima_p.data'
f = open(data_arima_p_file, 'rb')
data_arima_p = pickle.load(f)
print(data_arima_p)
print('预测长度为：',len(data_arima_p))



import matplotlib.pyplot as plt
# TODO： 绘图查看'新增确诊病例'后61行和data_arima_p的趋势（68-7=61）
data_s_ = df2[-len(data_arima_p):]  
ssss = pd.Series(data_arima_p,index = data_s_.index)
fig = plt.figure(figsize=(15,5))
plt.plot(ssss,color='blue', label='Predict')
plt.plot(data_s_,color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((data_s_.values - ssss.values)**2)/data_s_.size))
plt.show()



#函数5  特征抽取 (被函数6反复调用)
def feature_extraction(Series1,window=3): 
    # 生成空DataFrame
    df = pd.DataFrame()
    # 保留data_arima_p原值
    df['feature0'] = Series1.values
    # 滚动求取3天的均值、和、标准差、最大值、最小值
    df['feature1'] = Series1.rolling(window).mean().values
    df['feature2'] = Series1.rolling(window).sum().values
    df['feature3'] = Series1.rolling(window).std().values
    df['feature4'] = Series1.rolling(window).min().values
    df['feature5'] = Series1.rolling(window).max().values
    # 所有特征前一时点的取值也作为新特征
    df['feature6'] = df['feature0'].shift(1).values
    df['feature7'] = df['feature1'].shift(1).values
    df['feature8'] = df['feature2'].shift(1).values
    df['feature9'] = df['feature3'].shift(1).values
    df['feature10'] = df['feature4'].shift(1).values
    df['feature11'] = df['feature5'].shift(1).values
    df['feature12'] = df['feature6'].shift(1).values
    df['feature13'] = df['feature7'].shift(1).values
    df['feature14'] = df['feature8'].shift(1).values
    df['feature15'] = df['feature9'].shift(1).values
    df['feature16'] = df['feature10'].shift(1).values
    df['feature17'] = df['feature11'].shift(1).values
    # 以0填充缺失值
    df.fillna(0,inplace=True)
    return df



#函数6  获取最终数据
def get_fe_data(list1,origin_data,gap,testdays,lag=1,window=3):
    # 生成空列表
    data_get = [] #特征工程产生很多新的'y'作为新的'x'
    label_get = []
    # 以0填充data_arima_p缺少的那些天，并转为Series
    list1_0 = list(np.zeros(gap))
    list1_0.extend(list1)
    list1_1 = pd.Series(list1_0)
    # 切出用于生成标签的列
    label_list = origin_data['新增确诊病例']
    # 对每个测试日进行循环
    for i in range(testdays):
        # 将第0日至第i个测试日的数据使用函数5进行特征抽取
        ssss_1 = feature_extraction(list1_1.iloc[:i-testdays-lag+1],window) ###
        # 将第0日至第i个测试日的原始数据提出
        other_f = origin_data.iloc[:i-testdays-lag+1,:]
        # 调整index，并合并抽取的数据和原始数据
        ssss_1.index = other_f.index
        data_i = pd.concat([ssss_1,other_f],axis=1)
        # 储存特征DataFrame至列表data_get
        data_get.append(data_i)
        # 向后滚动'新增确诊病例'以获取标签，lag即为滚动距离
        label_ = label_list.shift(-lag-1)
        # 将第0日至第i个测试日的标签提出
        label_i = label_.iloc[:i-testdays-lag+1]
        # 储存标签Series至列表label_get
        label_get.append(label_i)
    return data_get,label_get



# 计算gap，并调用函数6，将特征和标签分别储存在features和lables中
gap = len(df1)-len(data_arima_p)
features,labels= get_fe_data(data_arima_p,df1,gap,20,1)



features



labels # label[-1]没有2020-03-07 #原来df2的41是在2020-03-07，现在在2020-03-05



# TODO；数据规整
from sklearn.preprocessing import StandardScaler
features_fe_tr_temp = []
features_fe_te_temp = []
labels_tr = []
labels_te = []
for i in range(len(features)):
    data_tr = features[i]
    X_train = data_tr.iloc[:-1,:]
    X_test = data_tr.iloc[-1:,:]
    y_train = labels[i].iloc[:-1]
    y_test = labels[i].iloc[-1:]
    
    stds = StandardScaler()
    X_train_std = stds.fit_transform(X_train)
    X_test_std = stds.transform(X_test)
    
    features_fe_tr_temp.append(X_train_std)
    features_fe_te_temp.append(X_test_std)
    labels_tr.append(y_train)
    labels_te.append(y_test)



len(features)



# TODO：特征过滤 (对新的'x'进行降维)
from sklearn.decomposition import PCA
features_fe_tr = []
features_fe_te = []

for i in range(len(features)):
    pca = PCA(6,random_state=0)
    X_train_pca = pca.fit_transform(features_fe_tr_temp[i])
    X_test_pca = pca.transform(features_fe_te_temp[i])
    
    features_fe_tr.append(X_train_pca)
    features_fe_te.append(X_test_pca)



# 选择模型进行交叉验证和网格搜索

# Gradient Boosting Decision Tree梯度提升决策树
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
# 无网格搜索调参（目的：查看参数原始状态）
gbdt = GradientBoostingRegressor()
X_tr_0 = features_fe_tr[-1]
y_tr_0 = labels_tr[-1]
gbdt.fit(X_tr_0,y_tr_0)



# TODO：网格搜索
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import time
best_params = pd.DataFrame(index = list(range(20)),columns = ['learning_rate','max_depth','n_estimators','score','消耗时间'])

#实例化模型
gbdt = GradientBoostingRegressor(loss='lad',random_state = 0)
#参数
params = {'learning_rate':[0.01,0.05,0.1,0.15],'max_depth':list(range(3,10,2)),'n_estimators':[60,90,120,150]} #按首字母排列
#实例化网格搜索
gs_gbdt = GridSearchCV(estimator = gbdt,param_grid = params,cv = 5,scoring = make_scorer(r2_score))
#滚动调参
def get_best_params(i):
    '''"i" passes an integer in range(20)'''
    start = time.time()
    X_tr = features_fe_tr[i]
    y_tr = labels_tr[i]
    gs_gbdt.fit(X_tr,y_tr)
    end = time.time()
    best_params.iloc[i,:-2] = gs_gbdt.best_params_ #返回结果按首字母排列
    best_params.iloc[i,-2] = gs_gbdt.best_score_
    best_params.iloc[i,-1] = '%.3f S'% float(end - start)

for j in range(20):
    get_best_params(j)

best_params



# TODO：模型选择、交叉验证
# 这里通过定义函数进行，四个参数从前向后分别为训练集和测试集的特征和标签
def gbdt_loop(xtr,xte,ytr,yte):  
    # 生成空列表储存预测值
    pred = []
    # 对数据集在循环中进行操作，tqdm生成进度条
    for i in tqdm(range(len(features))):
        # 提取相应数据数据
        X_tr = xtr[i]
        X_te = xte[i]
        y_tr = ytr[i]
        y_te = yte[i]
        # 使用GBDT，损失函数采取绝对值函数，更鲁棒
        gbdt = GradientBoostingRegressor(learning_rate = best_params.iloc[i,0],max_depth = int(best_params.iloc[i,1]),loss='lad',
                                         n_estimators = int(best_params.iloc[i,2]),random_state=0)
        # 拟合训练集
        gbdt.fit(X_tr,y_tr)
        # 预测测试集
        c = gbdt.predict(X_te)
        # 保存测试集的预测值
        pred.append(c[-1])
    return pred
pred = gbdt_loop(features_fe_tr,features_fe_te,labels_tr,labels_te)



pred



# TODO：模型评价
from sklearn.metrics import r2_score
r2score = r2_score(df1['新增确诊病例'][-len(pred):].values,pred[:])
fig = plt.figure(figsize=(15,5))
fig1 = fig.add_subplot(1,1,1)
fig1.plot(pred[:],label='pred')
fig1.plot(df1['新增确诊病例'][-len(pred):].values,label='label')
plt.title('R2_SCORE: %.4f'% r2score)
plt.xticks(range(20),labels=df1['新增确诊病例'][-len(pred):].index.date,rotation=30)
plt.legend()
plt.show()



# Random forest 随机森林模型
from sklearn.ensemble import RandomForestRegressor
# 无网格搜索调参（目的：查看参数原始状态）
rfr = RandomForestRegressor()
X_tr_0 = features_fe_tr[-1]
y_tr_0 = labels_tr[-1]
rfr.fit(X_tr_0,y_tr_0)



# TODO：网格搜索
best_params = pd.DataFrame(index = list(range(20)),columns = ['max_depth','max_features','max_leaf_nodes','n_estimators',\
                                                              'score','消耗时间'])
#实例化模型
rfr = RandomForestRegressor(random_state = 1)
#参数
params = {'max_depth':list(range(3,7,1)),'max_features':list(range(3,7,1)),'max_leaf_nodes':list(range(5,12,2)),\
          'n_estimators':list(range(5,12,2))} #按首字母排列
#实例化网格搜索
gs_rfr = GridSearchCV(estimator = rfr,param_grid = params,cv = 5,scoring = make_scorer(r2_score))
#滚动调参
def get_best_params(i):
    '''"i" passes an integer in range(20)'''
    start = time.time()
    X_tr = features_fe_tr[i]
    y_tr = labels_tr[i]
    gs_rfr.fit(X_tr,y_tr)
    end = time.time()
    best_params.iloc[i,:-2] = gs_rfr.best_params_ #返回结果按首字母排列
    best_params.iloc[i,-2] = gs_rfr.best_score_
    best_params.iloc[i,-1] = '%.3f S'% float(end - start)

for j in range(20):
    get_best_params(j)

best_params



# TODO：模型选择、交叉验证
# 这里通过定义函数进行，四个参数从前向后分别为训练集和测试集的特征和标签
def rfr_loop(xtr,xte,ytr,yte):  
    # 生成空列表储存预测值
    pred = []
    # 对数据集在循环中进行操作，tqdm生成进度条
    for i in tqdm(range(len(features))):
        # 提取相应数据数据
        X_tr = xtr[i]
        X_te = xte[i]
        y_tr = ytr[i]
        y_te = yte[i]
        # 使用随机森林
        rfr = RandomForestRegressor(max_depth = best_params.iloc[i,0],max_features = best_params.iloc[i,1],\
                                    max_leaf_nodes = best_params.iloc[i,2],n_estimators = best_params.iloc[i,3],random_state = 1)
        # 拟合训练集
        rfr.fit(X_tr,y_tr)
        # 预测测试集
        c = rfr.predict(X_te)
        # 保存测试集的预测值
        pred.append(c[-1])
    return pred
pred = rfr_loop(features_fe_tr,features_fe_te,labels_tr,labels_te)



pred



# TODO：模型评价
r2score = r2_score(df1['新增确诊病例'][-len(pred):].values,pred[:])
fig = plt.figure(figsize=(15,5))
fig1 = fig.add_subplot(1,1,1)
fig1.plot(pred[:],label='pred')
fig1.plot(df1['新增确诊病例'][-len(pred):].values,label='label')
plt.title('R2_SCORE: %.4f'% r2score)
plt.xticks(range(20),labels=df1['新增确诊病例'][-len(pred):].index.date,rotation=30)
plt.legend()
plt.show()



# Bagging
from sklearn.ensemble import BaggingRegressor
# 无网格搜索调参（目的：查看参数原始状态）
br = BaggingRegressor()
X_tr_0 = features_fe_tr[-1]
y_tr_0 = labels_tr[-1]
br.fit(X_tr_0,y_tr_0)



# TODO：网格搜索
best_params = pd.DataFrame(index = list(range(20)),columns = ['max_features','max_samples','n_estimators','score','消耗时间'])
#实例化模型
br = BaggingRegressor(random_state = 1)
#参数
params = {'max_features':list(range(4,8,1)),'max_samples':list(range(18,31,3)),'n_estimators':list(range(5,12,2))} #按首字母排列
#实例化网格搜索
gs_br = GridSearchCV(estimator = br,param_grid = params,cv = 5,scoring = make_scorer(r2_score))
#滚动调参
def get_best_params(i):
    '''"i" passes an integer in range(20)'''
    start = time.time()
    X_tr = features_fe_tr[i]
    y_tr = labels_tr[i]
    gs_br.fit(X_tr,y_tr)
    end = time.time()
    best_params.iloc[i,:-2] = gs_br.best_params_ #返回结果按首字母排列
    best_params.iloc[i,-2] = gs_br.best_score_
    best_params.iloc[i,-1] = '%.3f S'% float(end - start)

for j in range(20):
    get_best_params(j)

best_params



# TODO：模型选择、交叉验证
# 这里通过定义函数进行，四个参数从前向后分别为训练集和测试集的特征和标签
def br_loop(xtr,xte,ytr,yte):  
    # 生成空列表储存预测值
    pred = []
    # 对数据集在循环中进行操作，tqdm生成进度条
    for i in tqdm(range(len(features))):
        # 提取相应数据数据
        X_tr = xtr[i]
        X_te = xte[i]
        y_tr = ytr[i]
        y_te = yte[i]
        # 使用Bagging
        br = BaggingRegressor(max_features = best_params.iloc[i,0],max_samples = best_params.iloc[i,1],\
                              n_estimators = best_params.iloc[i,2],random_state = 1)
        # 拟合训练集
        br.fit(X_tr,y_tr)
        # 预测测试集
        c = br.predict(X_te)
        # 保存测试集的预测值
        pred.append(c[-1])
    return pred
pred = br_loop(features_fe_tr,features_fe_te,labels_tr,labels_te)

pred



# TODO：模型评价
r2score = r2_score(df1['新增确诊病例'][-len(pred):].values,pred[:])
fig = plt.figure(figsize=(15,5))
fig1 = fig.add_subplot(1,1,1)
fig1.plot(pred[:],label='pred')
fig1.plot(df1['新增确诊病例'][-len(pred):].values,label='label')
plt.title('R2_SCORE: %.4f'% r2score)
plt.xticks(range(20),labels=df1['新增确诊病例'][-len(pred):].index.date,rotation=30)
plt.legend()
plt.show()