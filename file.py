import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
data=pd.read_csv('covid.csv')
print(data.columns)
print(data.describe())
print(data.isna().sum())
print(data.memory_usage())
print(data.info())
print(data.dtypes)
col=data.columns.values
for i in col[1:]:
    print(data[i].info())
    print(data[i].describe())
    print(data[i].shape)
    print(data[i].isna())
    print(data[i].value_counts())
    #sn.boxplot(data[i])
    #plt.show()
sn.heatmap(data[col[1:]])
plt.show()
sn.countplot(data['state'].values[1:10])
plt.show()
sn.countplot(data['state'].values[10:20])
plt.show()
sn.countplot(data['state'].values[21:])
plt.show()
sn.pairplot(data[col[:-1]])
plt.show()
x=data[['confirmed','active','passive','deaths','dose1','dose2','dose3','precaution_dose','total_doses']]
y=data['population']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print(lr.score(x_test,y_test))
from sklearn.metrics import r2_score
print('r2_score in linear_regression is ',r2_score(y_test,pred))
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
mse=mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)
print('the mae is ',mean_absolute_error(y_test,pred))
print('the mse value is ',mse)
print('the rmse value is ',rmse)
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.metrics,keras.losses
model=Sequential()
model.add(Dense(units=x_train.shape[1],input_dim=x.shape[1],activation=keras.activations.relu))
model.add(Dense(units=x.shape[1],activation=keras.activations.relu))
model.add(Dense(units=x.shape[1],activation=keras.activations.relu))
model.add(Dense(units=x.shape[1],activation=keras.activations.relu))
model.add(Dense(units=1.,activation=keras.activations.relu))
model.compile(optimizer='adam',loss=keras.losses.mean_squared_error,metrics='mse')
model.fit(x_train,y_train,batch_size=20,epochs=200,verbose=False)
print('the r2_score of keras model is: ',r2_score(y_test,model.predict(x_test)))
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
ada=AdaBoostRegressor()
ada.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))
grda=GradientBoostingRegressor()
grda.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))
x=data[['dose1','dose2','dose3','precaution_dose','total_doses']]
y=data['population']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print(lr.score(x_test,y_test))
from sklearn.metrics import r2_score
print('r2_score in linear_regression is ',r2_score(y_test,pred))
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
ada=AdaBoostRegressor()
ada.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))
grda=GradientBoostingRegressor()
grda.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
print('The r2_score of ada boost is: ',r2_score(y_test,ada.predict(x_test)))