#to plot the values
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
# to import the dataset and read the data
import numpy as np
import pandas as pd
wd =pd.read_csv('C:/Users/Devna Chaturvedi/Desktop/Python Exam 1/question 3/weather.csv')
wd.head()
print(wd.head())
# to plot the correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(wd.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
wd.shape
print(wd.shape)
print(wd.info())
wd['Loud Cover'].value_counts()
wd.drop('Loud Cover',axis=1,inplace=True)
print(wd.shape)
# the loud cover axis consists of null values so will drop the column
rcParams['figure.figsize'] = 9, 9
wd.hist()
plt.show()

# Since the column of formatted date and daily summer is not essential it is dropped
wd.drop(['Formatted Date','Daily Summary'],axis=1,inplace=True)
wd.drop(['Wind Bearing (degrees)'],axis=1,inplace=True)
print(wd.shape)
wd.isnull().sum()
print(wd.isnull().sum())
print(wd['Precip Type'].value_counts())
wd['Precip Type'].fillna(method='ffill',inplace=True,axis=0)
wd['Precip Type'].value_counts()
wd.drop('Precip Type',axis=1,inplace=True)
pressure_median = wd['Pressure (millibars)'].median()

# pressure to plot the histogram since it cannot be zero as compared to other values like temperature which can be zero
def pressure(y):
    if y == 0:
        return y + pressure_median
    else:
        return y


wd["Pressure (millibars)"] = wd.apply(lambda row: pressure(row["Pressure (millibars)"]), axis=1)

rcParams['figure.figsize'] = 6, 4
wd['Pressure (millibars)'].hist()
plt.show()
# correlation between variables
# y is a dependent variable and x has independent variables
y=wd.iloc[:,0]
x=wd.iloc[:,2:]
print(x)
print(x.corr())
sns.regplot(x="Apparent Temperature (C)", y="Temperature (C)", data=wd);
plt.show()
# apparent temperature and temperature correlation is nearly equal to 1 so we remove the apparent temperature section
x.drop('Apparent Temperature (C)',axis=1,inplace=True)
x.shape
x_cols=x.columns
print(wd)
# training and testing the data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)

# normalize the dataset with respect to the independent variables
from sklearn.preprocessing import StandardScaler


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Classifying the model to check

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
# this part consists checking model
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
from sklearn import metrics
print('The accuracy is:')
print(metrics.accuracy_score(y_test,y_pred)*100)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
ls = linear_model.LinearRegression()
# the linear regression  model is used to check the correlation between temperature and other variables
# use the linear regression to determine the correlation between Humidity and temperature
X = wd["Humidity"].values.reshape(-1,1)
y = wd["Temperature (C)"].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print("Linear Regression")
ls.fit(X_train, y_train)

print("Calculating some regression quality metrics for humidity")
y_pred = ls.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
# use the linear regression to determine the correlation between pressure and temperature
X = wd["Pressure (millibars)"].values.reshape(-1,1)
y = wd["Temperature (C)"].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print("Linear Regression")
ls.fit(X_train, y_train)
print("Calculating some regression quality metrics for pressure")
y_pred = ls.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
# use the linear regression to determine the correlation between wind speed and temperature
X = wd["Visibility (km)"].values.reshape(-1,1)
y = wd["Temperature (C)"].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
print("Linear Regression")
ls.fit(X_train, y_train)
print("\n\nCalculating some regression quality metrics for visibility")
y_pred = ls.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
sns.regplot(x="Humidity", y="Temperature (C)", data=wd);
plt.show()
sns.regplot(x="Visibility (km)", y="Temperature (C)", data=wd);
plt.show()