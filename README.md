# weather-Forecasting-ML-project-Linear-regression-
#Comparing the maximum temperature and minimum temperature . This project would predict the max temperature.

# importing the library
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

# import the raw data
dataframe=pd.read_csv('C:/Users/user/Downloads/seattle-weather.csv')

print(dataframe.shape)

print(dataframe.describe())
dataframe.plot(x='temp_min' , y='temp_max', style='o')
plt.title('Min temperature vs Max temperature')
plt.xlabel('Min temperature')
plt.ylabel('Max temperature')
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.displot(dataframe['temp_max'])
plt.show()

# separate the train and test data
x=dataframe['temp_min'].values.reshape(-1,1)
y=dataframe['temp_max'].values.reshape(-1,1)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=0)

# Initialize the regression and train it
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# To retrive the intercept
print('Intercept:',regressor.intercept_)
print('Coefficient:',regressor.coef_)

# Show the Actual and predicted data
y_pred=regressor.predict(x_test)
df=pd.DataFrame({'Actual':y_test.flatten() , 'Predicted':y_pred.flatten()})
print(df)

df1=df.head(25)
df1.plot(kind='bar' , figsize=(16,10))
plt.grid(which='major', linestyle='-',linewidth='.5',color='green')
plt.grid(which='minor', linestyle=':',linewidth='.5',color='black')
plt.show()

# Real predictin graph
plt.scatter(x_test,y_test,color='gray')
plt.plot(x_test,y_pred,color='red',linewidth=2)
plt.show()

# measure the error
print('Mean absolute error', metrics.mean_absolute_error(y_test,y_pred))
print('Mean square error',metrics.mean_squared_error(y_test,y_pred))
print('Root mean square error',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

