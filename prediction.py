import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Demo_Data_Loan.csv')
# removing unnecessary columns
df.drop('Loan_ID', axis=1, inplace=True)
# check how many male and female
df.Gender.value_counts(dropna=False)
# check how many self employed
df.Self_Employed.value_counts(dropna=False)
# check eduaction
df.Education.value_counts(dropna=False)

# Replacing null values with mode

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
# replacing numerical null values with mean

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df.rename(columns={'Loan Approved': 'Loan_Approved'}, inplace=True)
df = pd.get_dummies(df)
# Drop unnecessary dummy columns
df = df.drop(['Gender_Female', 'Married_No', 'Education_Not Graduate',
              'Self_Employed_No'], axis=1)

# Rename columns name
new = {'Gender_Male': 'Gender', 'Married_Yes': 'Married',
       'Education_Graduate': 'Education', 'Self_Employed_Yes': 'Self_Employed'}

df.rename(columns=new, inplace=True)
df1 = df.copy()
X = df1.drop('Loan_Approved', axis=1)
y = df1['Loan_Approved']
X, y = SMOTE().fit_resample(X, y)
df2 = pd.concat([X, y], axis=1)
Q1 = df2.quantile(0.15)
Q3 = df2.quantile(0.85)
IQR = Q3 - Q1

df2 = df2[~((df2 < (Q1 - 1.5 * IQR)) | (df2 > (Q3 + 1.5 * IQR))).any(axis=1)]
df2.ApplicantIncome = np.sqrt(df2.ApplicantIncome)
df2.CoapplicantIncome = np.sqrt(df2.CoapplicantIncome)
df2.LoanAmount = np.sqrt(df2.LoanAmount)

X = df2.drop('Loan_Approved', axis=1)
y = df2['Loan_Approved']

X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

KNclassifier = KNeighborsClassifier(n_neighbors=10)
KNclassifier.fit(X_train, y_train)

y_pred = KNclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score

KNNAcc = accuracy_score(y_pred, y_test)
print('KNN best accuracy: {:.2f}%'.format(KNNAcc * 100))

RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=23)
RFclassifier.fit(X_train, y_train)

y_pred = RFclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score

RFAcc = accuracy_score(y_pred, y_test)
print('Random Forest accuracy: {:.2f}%'.format(RFAcc * 100))

y_pred = RFclassifier.predict([[1200, 0, 110.0, 180, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]])
print(y_pred)

SVCclassifier = SVC(kernel='rbf', max_iter=500)
SVCclassifier.fit(X_train, y_train)

y_pred = SVCclassifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score

SVCAcc = accuracy_score(y_pred, y_test)
print('SVC accuracy: {:.2f}%'.format(SVCAcc * 100))


import pickle

with open('RF_model_loan', 'wb') as file:
    pickle.dump(RFclassifier, file)


