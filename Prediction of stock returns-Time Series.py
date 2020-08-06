#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from statsmodels.tsa.seasonal import seasonal_decompose


# In[3]:


from nsepy import get_history
from datetime import datetime
import matplotlib.pyplot as plt
from nsepy import get_index_pe_history
from nsetools import Nse
nse=Nse()
print(nse)
from nsepy.symbols import get_symbol_list


# # Step-1 : Download stock price from 01-03-2019 to 01-03-2020

# In[4]:


start=datetime(2019,3,1)
end=datetime(2020,3,1)


# In[5]:


stock=get_history(symbol='HDFC',start=start,end=end)


# # Step-2: Calculate trend and seasonality of Close,Low and High Price

# In[6]:


ts_ds=pd.DataFrame({'Symbol':'HDFC','Low':stock[stock['Symbol']=='HDFC'].Low,'Low_season':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].Low),freq=90,extrapolate_trend='freq').seasonal,'Low_trend':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].Low),freq=90,extrapolate_trend='freq').trend,
                   'High':stock[stock['Symbol']=='HDFC'].High,'High_season':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].High),model='multiplicative',freq=90,extrapolate_trend='freq').seasonal,'High_trend':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].High),freq=90,extrapolate_trend='freq').trend,
                   'Close':stock[stock['Symbol']=='HDFC'].Close,'Close_season':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].Close),model='multiplicative',freq=90,extrapolate_trend='freq').seasonal,'Close_trend':seasonal_decompose(np.array(stock[stock['Symbol']=='HDFC'].Close),freq=90,extrapolate_trend='freq').trend})
ts_ds['Low_trend_season']=ts_ds['Low_season']+ts_ds['Low_trend']
ts_ds['High_trend_season']=ts_ds['High_season']+ts_ds['High_trend']
ts_ds['Close_trend_season']=ts_ds['Close_season']+ts_ds['Close_trend']


# In[7]:


ts_ds.info()


# # Step-3: Splitting into test and train data

# In[8]:


x_low=pd.DataFrame(sm.add_constant(ts_ds['Low_trend_season'])).reset_index()
y_low=pd.DataFrame(ts_ds['Low']).reset_index()


# In[9]:


train_x_low,test_x_low,train_y_low,test_y_low=train_test_split(x_low['Low_trend_season'],y_low['Low'],train_size=0.8,random_state=42)


# # Step-3(a):Regression between trend/seasonality and Low price

# In[10]:


model_low=sm.OLS(train_y_low,train_x_low).fit()


# In[11]:


model_low.summary2()


# In[12]:


pred_y_low=model_low.predict(test_x_low)


# In[13]:


r2_low=metrics.r2_score(test_y_low,pred_y_low)
print('r_squared_low_test',r2_low)


# In[14]:


rmse=round(np.sqrt(r2_low),2)
print('RMSE_test',rmse)


# In[15]:


rmse_train=round(np.sqrt(metrics.mean_squared_error(train_y_low,train_x_low)))
print('RMSE_train',rmse_train)


# # Prediction on train data

# In[16]:


y_pred_train=pd.DataFrame({'Actual_low':train_y_low,'Predicted_value':model_low.predict(train_x_low)})


# In[17]:


plt.plot(y_pred_train.Actual_low,label='Actual price',color='b')
plt.plot(y_pred_train.Predicted_value,label='Predicted Price',color='r')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.show()


# # Prediction on test data

# In[18]:


y_pred_test=pd.DataFrame({'Actual_low':test_y_low,'Predicted_value':model_low.predict(test_x_low)})


# In[19]:


plt.plot(y_pred_test.Actual_low,label='Actual price',color='b')
plt.plot(y_pred_test.Predicted_value,label='Predicted Price',color='r')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.show()

