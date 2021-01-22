import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from scipy.stats import chi2
warnings.filterwarnings("ignore")
#%%
def autocorr_cal(ts,k):
    sum_top = 0
    variance = 0
    l = len(ts)
    ts1= ts[0:l-k]
    ts2 = ts[k:]
    ts_mean = ts.mean()
    for i in range(l):
        variance += (ts[i] - ts_mean)** 2
    for i in range(l-k):
        sum_top += (ts1[i]-ts_mean)*(ts2[i]-ts_mean)
    acf=sum_top/variance
    return acf
def plot_autocorr_cal(acf_series,lag):
    k_series = np.array(range(-lag+1,lag))
    plt.figure(figsize=(20, 6))
    plt.stem(k_series,acf_series, use_line_collection=True)
    plt.title('ACF', fontsize=25)
    plt.xlabel('Lag', fontsize=20)
    plt.show()
    return  acf_series

def GPAC_call(y,acf_series):
    y_var = np.var(y)
    ry=[]
    for i in acf_series:
        ry.append(i*y_var)
    #print(ry)
    phi=[]
    phi_1=[]
    i=0
    gpac = np.zeros(shape=(8, 7))
    for j in range(0,8):
        for k in range(2,9):
            bottom = np.zeros(shape=(k, k))
            top = np.zeros(shape=(k, k))
            for m in range(k):  ##row
                for n in range(k):  ##column
                    bottom[m][n]=ry[abs(j+m - n)]
                top[m][-1]=ry[abs(j+m+1)]

            #print('bottom\n',bottom)
            i=i+1
            top[:,:k-1] = bottom[:,:k-1]
            #print('top\n',top)

            phi.append(round((np.linalg.det(top) / np.linalg.det(bottom)),2))
        phi_1.append(round(ry[j + 1] / ry[j],2))

    gpac=np.array(phi).reshape(8,7)

    Phi1=pd.DataFrame(phi_1)
    Gpac=pd.DataFrame(gpac)
    GPAC = pd.concat([Phi1,Gpac], axis=1)
    GPAC.columns=['1','2','3','4','5','6','7','8']
    print(GPAC)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(GPAC)
    plt.title("Generalized partial autocorrelation function ")
    plt.xlabel("na")
    plt.ylabel("nb")
    plt.show()

def q_value_call(acf_series,error,h):
    sum_acf_square = 0
    for i in range(h):
        sum_acf_square += acf_series[i+1] **2
    Q = len(error)*sum_acf_square
    return Q
#%% d. Preprocessing procedures
data = pd.read_csv('train.csv',index_col='Date',header=0)

print('Check missing value:\n',data.isnull().sum())
store1=data[(data.Store ==1) & (data.Open==1)].sort_values(by = ['Date'],ascending = True)
store1_stateholiday = store1[(store1.StateHoliday=='1')]
store1_zero=store1[(store1.Sales==0)]

#%% 3.a Plot of the dependent variable versus time.
store1=store1.reset_index(drop=False)
fig = plt.figure(figsize=(20,8))
ax = plt.subplot()
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.plot(store1['Date'],store1['Sales'])
ax.set_title('Sales verus time ',fontsize=25)
plt.show()
#%%
diff_1 = store1['Sales'].diff(360)
diff_1.dropna(inplace=True)
diff_1.plot(kind='line',linewidth=2,figsize=(16,8))
plt.title('Sales verus time After 1st Difference Transformation')
plt.ylabel("Sales")
plt.show()
#%% 3.b ACF of the dependent variable.
Sales=store1['Sales']

lag=50
acf_series_right=[]
acf_series_left=[]
acf_series = []
for i in range(lag):
    acf_series_right.append(autocorr_cal(np.array(Sales),i))
acf_series_left= list(reversed(acf_series_right))
acf_series = np.concatenate((acf_series_left,acf_series_right[1:]))

plot_autocorr_cal(acf_series,lag)

#%% 3.e Split the dataset into train set (80%) and test set (20%).
#
X=store1.drop(['Open','Store','Sales','StateHoliday'],axis = 1)

y=store1['Sales']
y=np.mat(np.asarray(y).reshape(-1,1))
X_train_raw, X_test_raw,y_train, y_test= train_test_split(X,y,shuffle=False,test_size=0.2)

X_train=X_train_raw.set_index('Date',drop=True)
X_test=X_test_raw.set_index('Date',drop=True)
#print(X_train)
#print(X_test)
X_train=np.asarray(X_train).reshape(-1,4)

X_test=np.asarray(X_test).reshape(-1,4)
#print(X_train)
#print(X_test)
X_train=np.column_stack((np.ones(len(X_train)),X_train))
X_train=np.mat(X_train)
X_test=np.column_stack((np.ones(len(X_test)),X_test))
X_test=np.mat(X_test)
#print(y)

#%% 4. stationary test-ADF
result = adfuller(store1['Sales'])
print("ADF test for Sales of store1:")
print("ADF Statistic: %f" %result[0])
print("p-value: %f" %result[1])
print("Critical Value:")
for key, value in result[4].items():
    print('\t%s: %.3f' %(key,value))

#%% 5. Time series Decomposition
###Q5 seasonal_decompose
# additvie
add_decomposition = seasonal_decompose(Sales, model='additive',freq=300)
add_decomposition.plot().suptitle('Addictive Decomposition')
plt.figure(figsize=(20, 16))
plt.show()

#multiplicative
mul_decomposition = seasonal_decompose(Sales, model='multiplicative',freq=360)
mul_decomposition.plot().suptitle('Multiplicative Decomposition')
plt.figure(figsize=(16, 8))
plt.show()

#%% 6. Holt-Winters method:
#print("Holt-Winter Seasonal Method")
#print("X_test_raw",X_test_raw)
y_train=np.array(y_train).astype('double')
model = ets.ExponentialSmoothing(y_train,trend='multiplicative',seasonal='multiplicative',seasonal_periods=360).fit()

y_predict = model.predict(start=1,end=len(y))

Date_train=np.asarray(X_train_raw['Date'])
Date_test=np.asarray(X_test_raw['Date'])

error_holt=y_test-y_predict[len(y_train):]

mean_error=np.mean(error_holt)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))

var_error = np.var(error_holt)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))

mse=np.mean(np.square(error_holt))
print('the MSE of the residualsis:','{0:0.4f}'.format(mse))

rmse=np.sqrt(mse)
print('the RMSE of the residualsis:','{0:0.4f}'.format(rmse))

mape=np.mean(np.abs(error_holt)/np.abs(y_test))
print('the MAPE of the residualsis:','{0:0.4f}'.format(mape))


fig, ax1 = plt.subplots(figsize=(20, 10))
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3= fig.add_subplot(111)
ax1.plot(Date_train,y_train)
ax2.plot(Date_test,y_test)
ax3.plot(Date_test,y_predict[len(y_train):])
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.title("Prediction Based on Holt-Winters Method")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend(['Train','Test','Predict'],loc='upper left')
plt.show()
#%% 7. Feature selection:
corr_all = store1.drop(['Open','Store'],axis = 1).corr()
ax=sns.heatmap(corr_all,cmap="YlGnBu")
plt.title('Heatmap')
plt.show()
sns.pairplot(corr_all)
plt.show()

#%% 8.Develop the multiple linear regression

XT_X = np.dot(X_train.T,X_train)# the same as X_train*X_train.T

beta_hat = np.dot(np.dot(XT_X.I,X_train.T),y_train)

print('beta_hat',beta_hat)

X_train=X_train[:,1:]
X_test = X_test[:,1:]

y_predict = beta_hat[0]+X_test*beta_hat[1:]

y_predict =np.ravel(y_predict)
y_test = np.ravel(y_test)
#%%
store1=store1.set_index('Date',drop=True)
X=store1.drop(['Open','Store','Sales','StateHoliday'],axis = 1)

X=np.asarray(X).reshape(-1,4)
X=np.column_stack((np.ones(len(X)),X))
X=np.mat(X)
y=store1['Sales']
y=np.mat(np.asarray(y).reshape(-1,1))

X_train, X_test,y_train, y_test= train_test_split(X,y,shuffle=False,test_size=0.2)
model = sm.OLS(y_train,X_train).fit()
y_predict = model.predict(X_test)
print(model.summary())

error_linear1 = y_predict-y_test

mse=np.mean(np.square(error_linear1))
print('the mean square error of the error of the residualsis:','{0:0.4f}'.format(mse))


mean_error=np.mean(error_linear1)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))

var_error = np.var(error_linear1)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))
#%%
X = store1.drop(['Open','Store','Sales','StateHoliday','DayOfWeek','SchoolHoliday'],axis = 1)

X=np.asarray(X).reshape(-1,2)
X=np.column_stack((np.ones(len(X)),X))
X=np.mat(X)
y=store1['Sales']
y=np.mat(np.asarray(y).reshape(-1,1))

X_train, X_test,y_train, y_test= train_test_split(X,y,shuffle=False,test_size=0.2)
model = sm.OLS(y_train,X_train).fit()
print(model.summary())
y_predict = model.predict(X_test)
y_test=np.ravel(y_test)

error_linear2 = y_predict-y_test
mean_error=np.mean(error_linear2)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))
#print('error_linear2',error_linear2)
var_error = np.var(error_linear2)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))

mse=np.mean(np.square(error_linear2))
print('the MSE of the residualsis:','{0:0.4f}'.format(mse))

rmse=np.sqrt(mse)
print('the RMSE of the residualsis:','{0:0.4f}'.format(rmse))

mape=np.mean(np.abs(error_linear2)/np.abs(y_test))
print('the MAPE of the residualsis:','{0:0.4f}'.format(mape))

#print('error\n',error)
lag=len(error_linear2)
acf_series_right=[]
acf_series_left=[]
acf_series = []
for i in range(lag):
    acf_series_right.append(autocorr_cal(np.array(error_linear2),i))
acf_series_left= list(reversed(acf_series_right))
acf_series = np.concatenate((acf_series_left,acf_series_right[1:]))

plot_autocorr_cal(acf_series,lag)

fig, ax1 = plt.subplots(figsize=(20, 10))
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3= fig.add_subplot(111)
ax1.plot(Date_train,y_train)
ax2.plot(Date_test,y_test)
ax3.plot(Date_test,y_predict)
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.title("Prediction Based on Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Train','Test','Predict'],loc='upper left')
plt.show()



#%% 9.ARMA model
#a
sales=np.asarray(Sales)
lag=20
acf_series = []
for i in range(lag):
    acf_series.append(autocorr_cal(sales, i))
#print('acf_series',acf_series)
GPAC_call(sales,acf_series)
#%%b na=3 nb=2
na=3
nb=2
model = sm.tsa.ARMA(sales,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i),"is:",'{0:0.4f}'.format(model.params[i]))
for i in range(nb):
    print("The MA coefficient b{}".format(i),"is:",'{0:0.4f}'.format(model.params[i+na]))
print(model.summary())
print('The root of an:',np.roots([0.8209,0.8276,-0.6485]))
print('The root of bn:',np.roots([-0.1841,-0.8109]))
interval = model.conf_int(alpha=0.05, cols=None)
print("the confidence interval for the estimated parameter(s) is:\n",interval)

cov_matrix = model.cov_params()
print("the estimated covariance matrix of the estimated parameters:\n",cov_matrix)


y_hat = model.predict(start=1,end=len(sales))
error1_all=sales-y_hat
error1=error1_all[len(y_train):]

mean_error=np.mean(error1)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))

var_error = np.var(error1)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))

mse=np.mean(np.square(error1))
print('the MSE of the residualsis:','{0:0.4f}'.format(mse))

rmse=np.sqrt(mse)
print('the RMSE of the residualsis:','{0:0.4f}'.format(rmse))

mape=np.mean(np.abs(error1)/np.abs(y_test))
print('the MAPE of the residualsis:','{0:0.4f}'.format(mape))

fig, ax1 = plt.subplots(figsize=(20, 10))
#plt.figure(figsize=(20,8))
#fig = plt.figure()
#ax2 = ax1.twinx()
#ax3 = ax1.twinx()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3= fig.add_subplot(111)
ax1.plot(Date_train,y_train)
ax2.plot(Date_test,y_test)
ax3.plot(Date_test,y_hat[len(y_train):])
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.title("Prediction Based on ARMA(3,2)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Train','Test','Predict'],loc='upper left')
plt.show()
#%%
#print('error\n',error)
lag=len(error1)
acf_series_right=[]
acf_series_left=[]
acf_series = []

for i in range(lag):
    acf_series_right.append(autocorr_cal(np.array(error1),i))
acf_series_left= list(reversed(acf_series_right))
acf_series = np.concatenate((acf_series_left,acf_series_right[1:]))
plot_autocorr_cal(acf_series,lag)

lag_q=10
h=lag_q
Q_value= q_value_call(acf_series_right,error1,h-1)
print("The Q value for this estimate is =",'{0:0.4f}'.format(Q_value))

DOF = lag_q - na -nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa,DOF)
print('chi_critical',chi_critical)
if Q_value<chi_critical:
    print("The residual is white")
if Q_value>chi_critical:
    print("The residual is not white")

#%%b na=1 nb=1
na=1
nb=1
model = sm.tsa.ARMA(sales,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i),"is:",'{0:.4f}'.format(model.params[i]))
for i in range(nb):
    print("The MA coefficient b{}".format(i),"is:",'{0:.4f}'.format(model.params[i+na]))
print(model.summary())

interval = model.conf_int(alpha=0.05, cols=None)
print("the confidence interval for the estimated parameter(s) is:\n",interval)

cov_matrix = model.cov_params()
print("the estimated covariance matrix of the estimated parameters:\n",cov_matrix)

y_hat = model.predict(start=1,end=len(sales))
error2_all=sales-y_hat
error2=error2_all[len(y_train):]

mean_error=np.mean(error2)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))

var_error = np.var(error2)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))

mse=np.mean(np.square(error2))
print('the MSE of the residualsis:','{0:0.4f}'.format(mse))

rmse=np.sqrt(mse)
print('the RMSE of the residualsis:','{0:0.4f}'.format(rmse))

mape=np.mean(np.abs(error2)/np.abs(y_test))
print('the MAPE of the residualsis:','{0:0.4f}'.format(mape))

fig, ax1 = plt.subplots(figsize=(20, 10))
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3= fig.add_subplot(111)
ax1.plot(Date_train,y_train)
ax2.plot(Date_test,y_test)
ax3.plot(Date_test,y_hat[len(y_train):])
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.title("Prediction Based on ARMA(1,1)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Train','Test','Predict'],loc='upper left')
plt.show()
#%%

lag=len(error2)
acf_series_right=[]
acf_series_left=[]
acf_series = []

for i in range(lag):
    acf_series_right.append(autocorr_cal(np.array(error2),i))
acf_series_left= list(reversed(acf_series_right))
acf_series = np.concatenate((acf_series_left,acf_series_right[1:]))
plot_autocorr_cal(acf_series,lag)

lag_q=5
h=lag_q
Q_value= q_value_call(acf_series_right,error1,h-1)
print("The Q value for this estimate is =",'{0:0.4f}'.format(Q_value))

DOF = lag_q - na -nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa,DOF)
print('chi_critical',chi_critical)
if Q_value<chi_critical:
    print("The residual is white")
if Q_value>chi_critical:
    print("The residual is not white")
#%%ARIMA
#!pip install pmdarima
model_autoARIMA = auto_arima(sales, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

#%%
#%%b na=2 nb=3
na=2
nb=3
model = sm.tsa.ARMA(sales,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print("The AR coefficient a{}".format(i),"is:",'{0:.4f}'.format(model.params[i]))
for i in range(nb):
    print("The MA coefficient b{}".format(i),"is:",'{0:.4f}'.format(model.params[i+na]))
print(model.summary())

interval = model.conf_int(alpha=0.05, cols=None)
print("the confidence interval for the estimated parameter(s) is:\n",interval)

cov_matrix = model.cov_params()
print("the estimated covariance matrix of the estimated parameters:\n",cov_matrix)

y_hat = model.predict(start=1,end=len(sales))
error3_all=sales-y_hat
error3=error3_all[len(y_train):]

mean_error=np.mean(error2)
print('the estimated mean of error is:','{0:0.4f}'.format(mean_error))

var_error = np.var(error2)
print('the estimated variance of error is:','{0:0.4f}'.format(var_error))

mse=np.mean(np.square(error2))
print('the MSE of the residualsis:','{0:0.4f}'.format(mse))

rmse=np.sqrt(mse)
print('the RMSE of the residualsis:','{0:0.4f}'.format(rmse))

mape=np.mean(np.abs(error3)/np.abs(y_test))
print('the MAPE of the residualsis:','{0:0.4f}'.format(mape))

fig, ax1 = plt.subplots(figsize=(20, 10))
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3= fig.add_subplot(111)
ax1.plot(Date_train,y_train)
ax2.plot(Date_test,y_test)
ax3.plot(Date_test,y_hat[len(y_train):])
ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.title("Prediction Based on ARMA(1,1)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(['Train','Test','Predict'],loc='upper left')
plt.show()