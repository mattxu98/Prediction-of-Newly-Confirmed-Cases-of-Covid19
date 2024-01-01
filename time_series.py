# 1. 提取时间序列
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_excel('湖北.xlsx')
df1



df1['下日新增确诊'] = list(df1['新增确诊病例'][1:])+[np.nan]
df = pd.DataFrame(df1['下日新增确诊'][:-1])
df.set_index(df1['公开时间'][:-1],inplace = True)
df


# 2. 单位根检验(ADF检验) 和 白噪声检验
from statsmodels.tsa.stattools import adfuller #ADF检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf #画图定阶
from statsmodels.tsa.arima_model import ARIMA #模型
from statsmodels.tsa.arima_model import ARMA #模型

print(adfuller(df))



print(acorr_ljungbox(df, lags=1))



# 3. AIC、BIC定阶
import statsmodels.api as sm 

def get_pq(df):
    #AIC
    AIC = sm.tsa.arma_order_select_ic(df, max_ar=6, max_ma=4, ic='aic')['aic_min_order']
    #BIC
    BIC = sm.tsa.arma_order_select_ic(df, max_ar=6, max_ma=4, ic='bic')['bic_min_order']
    print('the AIC is{},\nthe BIC is{}\n'.format(AIC,BIC))
get_pq(df)



# 4. ARMA模型拟合与预测
model=ARMA(df,(1,1))
result_arma = model.fit(disp=-1)

predict_df = result_arma.predict()
predict_df



import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(predict_df ,label="forecast")
plt.plot(df,label="real")
plt.xlabel('Date',fontsize=12,verticalalignment='top')
plt.ylabel('Diff',fontsize=12,horizontalalignment='center')
plt.legend()
plt.show()



forecast_df = result_arma.forecast(5)
forecast_df