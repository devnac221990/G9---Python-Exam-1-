import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('creditcard.csv') # imports data





FraudSamples = dataset[dataset['Class'] == 1] # Recovery of fraud data
plt.figure(figsize=(15,10))
plt.scatter(FraudSamples['Time'], FraudSamples['Amount']) # Display fraud amounts according to their time
plt.title('Scratter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,200000])
plt.ylim([0,3000])
plt.show()
print("Fraud Samples:", len(FraudSamples))

LegitSamples = dataset[dataset['Class'] == 0] # Recovery of fraud data
plt.figure(figsize=(15,10))
plt.scatter(LegitSamples['Time'], LegitSamples['Amount']) # Display fraud amounts according to their time
plt.title('Scratter plot amount Legit')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,200000])
plt.ylim([0,3000])
plt.show()
print("Legit Samples:", len(LegitSamples))


X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state= 0)



classifier = SVC(kernel="linear") # Fitting Linear SVM to the Training set
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test) # Predicting the Test set results

# Calculating SVM Accuracy
print("SVM Accuracy is: ",metrics.accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))


### Balancing with undersampling


balancedDataset= dataset[:] #Copy the dataset
trainFraud=balancedDataset[balancedDataset['Class']==1] #only Fraud Values
trainLegit=balancedDataset[balancedDataset['Class']==0] #only Legit Transactions
# We need to mix the Data 50:50 Frauds and non Frauds
smallLegitTrSample=trainLegit.sample(len(trainFraud))  # Take only that Fraud amount of non Fraud transactions
trainMixData=(trainFraud.append(smallLegitTrSample)).sample(frac=1) #Mix them

#Start training the Mixed dataset
X = trainMixData.drop('Class', axis=1)
y = trainMixData['Class']

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state= 0)



classifier = SVC(kernel="linear") # Fitting Linear SVM to the Training set
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test) # Predicting the Test set results

print("SVM Accuracy is: ",metrics.accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))
