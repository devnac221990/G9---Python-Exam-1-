# Task 5
import random
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def filterMoney(m):
    if type(m) == str:
        m = m.replace('€', '')
        if 'K' in m:
            m = float(m.replace('K', '')) * 1000
        elif 'M' in m:
            m = float(m.replace('M', '')) * 1000000
        elif m.isalnum():
            m = float(m)
        return m
    elif type(m) in (int, float):
        return float(m)
    else:
        return 0.0


def convertHeight(h):
    if type(h) == str:
        splitHeight = h.split("'")

        feet = int(splitHeight[0])
        inches = int(splitHeight[1])
        return feet + (inches / 12)
    if type(h) == float:
        return h


def convertWeight(w):
    if 'lbs' in str(w):
        return int(w.replace('lbs', ''))
    elif type(w) == float:
        return w


def convertPreferredFoot(leg):
    if type(leg) == str:
        leg = leg.replace(' ', '')
        leg = leg.replace('\n', '')
        leg = leg.lower()
        if leg == 'right':
            return 1
        elif leg == 'left':
            return 0

    return random.randint(0, 1)


def convertContractValidUntil(y):
    if str(y).isalnum():
        return int(y)
    elif '-' in str(y):
        return int('20' + str(y[-2]) + str(y[-1]))
    return 2020


df = pd.read_csv('FifaStats.csv')  # imports data
df = df.drop(['Rem', 'ID', 'Name', 'Photo', 'Flag', 'Club Logo', 'Work Rate', 'Real Face', \
              'Jersey Number', 'Loaned From', 'Body Type', 'Joined', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', \
              'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', \
              'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Club', 'Nationality', 'Position'], axis=1)

df = df.dropna()
df['Age'] = df['Age'].apply(float)

df['Wage'] = df['Wage'].apply(filterMoney)
df['Value'] = df['Value'].apply(filterMoney)
df['Release Clause'] = df['Release Clause'].apply(filterMoney)
df['Height'] = df['Height'].apply(convertHeight)
df['Weight'] = df['Weight'].apply(convertWeight)
df['Preferred Foot'] = df['Preferred Foot'].apply(convertPreferredFoot)
df['Contract Valid Until'] = df['Contract Valid Until'].apply(convertContractValidUntil)

print(df.dtypes)


data = df.select_dtypes(include=[np.number]).interpolate().dropna()

corr = data.corr()
plt.figure(figsize=(20, 20));
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.show();
print(corr['Value'].sort_values(ascending=False)[1:11], '\n')
print('--------------------')
print(corr['Wage'].sort_values(ascending=False)[1:11], '\n')
print('--------------------')
print(corr['Potential'].sort_values(ascending=False)[1:11], '\n')

################################################################


X = data[['Overall', 'Release Clause', 'Value', 'Reactions', 'Wage', 'Special', 'International Reputation']]
y = data['Potential'].values.astype(int)

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating Accuracy
print("Naive Bayes accuracy is: ", metrics.accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))

############################################################
dataset = df.apply(LabelEncoder().fit_transform)

X_train = dataset[['Overall', 'Release Clause', 'Value', 'Reactions', 'Wage', 'Special', 'International Reputation']]
Y_train = dataset["Potential"].astype(int)
X_test = dataset[
    ['Overall', 'Release Clause', 'Value', 'Reactions', 'Wage', 'Special', 'International Reputation']].copy()

##KNN
knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN accuracy is:", acc_knn)

from sklearn.decomposition import PCA

X = dataset[['Overall', 'Release Clause', 'Value', 'Reactions', 'Wage', 'Special', 'International Reputation']]
y = dataset['Potential'].values

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = SVC(gamma='auto')  # Fitting rbf SVM to the Training set
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)  # Predicting the Test set results

# Calculating SVC Accuracy
print("RBF SVM Accuracy is: ", metrics.accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))

X = dataset[['Overall', 'Release Clause', 'Value', 'Reactions', 'Wage', 'Special', 'International Reputation']]
y = dataset['Potential'].values

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = SVC(kernel='linear')  # Fitting rbf SVM to the Training set
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)  # Predicting the Test set results

# Calculating SVC Accuracy
print("Linear SVM Accuracy is: ", metrics.accuracy_score(y_test, y_pred) * 100)
print(classification_report(y_test, y_pred))
