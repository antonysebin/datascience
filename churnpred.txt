1. Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
df.shape                                                           #churnprediction in telecom sector
#(7043, 21)



df.isna().sum().sum()
#0


df.Churn.value_counts()
No     5174
Yes    1869


columns = df.columns
binary_cols = []
for col in columns:
    if df[col].value_counts().shape[0] == 2:
        binary_cols.append(col)



fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
sns.countplot("gender", data=df, ax=axes[0,0])
sns.countplot("SeniorCitizen", data=df, ax=axes[0,1])
sns.countplot("Partner", data=df, ax=axes[0,2])
sns.countplot("Dependents", data=df, ax=axes[1,0])
sns.countplot("PhoneService", data=df, ax=axes[1,1])
sns.countplot("PaperlessBilling", data=df, ax=axes[1,2])


churn_numeric = {'Yes':1, 'No':0}
df.Churn.replace(churn_numeric, inplace=True)

df[['gender','Churn']].groupby(['gender']).mean()



table = pd.pivot_table(df, values='Churn', index=['gender'],
                    columns=['SeniorCitizen'], aggfunc=np.mean)


sns.countplot("InternetService", data=df)


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
sns.countplot("StreamingTV", data=df, ax=axes[0,0])
sns.countplot("StreamingMovies", data=df, ax=axes[0,1])
sns.countplot("OnlineSecurity", data=df, ax=axes[0,2])
sns.countplot("OnlineBackup", data=df, ax=axes[1,0])
sns.countplot("DeviceProtection", data=df, ax=axes[1,1])
sns.countplot("TechSupport", data=df, ax=axes[1,2])


df.PhoneService.value_counts()
Yes    6361
No      682
df.MultipleLines.value_counts()
No                  3390
Yes                 2971
No phone service     682


plt.figure(figsize=(10,6))
sns.countplot("Contract", data=df)


plt.figure(figsize=(10,6))
sns.countplot("PaymentMethod", data=df)


fig, axes = plt.subplots(1,2, figsize=(12, 7))
sns.distplot(df["tenure"], ax=axes[0])
sns.distplot(df["MonthlyCharges"], ax=axes[1])

df.drop([‘customerID’,’gender’,’PhoneService’,’Contract’,’TotalCharges’], axis=1, inplace=True)

2. Data Preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

cat_features = ['SeniorCitizen', 'Partner', 'Dependents',
'MultipleLines', 'InternetService','OnlineSecurity'      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
X = pd.get_dummies(df, columns=cat_features, drop_first=True)


sc = MinMaxScaler()
a = sc.fit_transform(df[['tenure']])
b = sc.fit_transform(df[['MonthlyCharges']])
X['tenure'] = a
X['MonthlyCharges'] = b
X.shape
(7043, 26)

Resampling

sns.countplot('Churn', data=df).set_title('Class Distribution Before Resampling')


X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]

X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
print(len(X_yes_upsampled))
5174


X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)
sns.countplot('Churn', data=X_upsampled).set_title('Class Distribution After Resampling')


3. Model Creation and Evaluation
from sklearn.model_selection import train_test_split
X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)
y = X_upsampled['Churn'] #target (dependent variable)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

clf_ridge = RidgeClassifier() #create a ridge classifier object
clf_ridge.fit(X_train, y_train) #train the model

pred = clf_ridge.predict(X_train)
accuracy_score(y_train, pred)
0.7574293307562213

pred_test = clf_ridge.predict(X_test)
accuracy_score(y_test, pred_test)
0.7608695652173914

Random Forest

from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_forest.fit(X_train, y_train)

pred = clf_forest.predict(X_train)
accuracy_score(y_train, pred)
0.8860835950712732
pred_test = clf_forest.predict(X_test)
accuracy_score(y_test, pred_test)
0.842512077294686

4. Improving the Model


from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}
forest = RandomForestClassifier()
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)


clf.fit(X, y)


clf.best_params_
{'max_depth': 20, 'n_estimators': 150}
clf.best_score_
0.8999806725937379



