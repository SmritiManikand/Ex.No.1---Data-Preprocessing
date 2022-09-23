# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```python
import pandas as pd
import numpy as np
df = pd.read_csv("/content/Churn_Modelling.csv")
df.info()
df.isnull().sum()
df.duplicated()
df.describe()
df['Exited'].describe()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df.copy()
df1["Geography"] = le.fit_transform(df1["Geography"])
df1["Gender"] = le.fit_transform(df1["Gender"])
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]] = pd.DataFrame(scaler.fit_transform(df1[["CreditScore","Geography","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]]))
df1
df1.describe()
X = df1[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]].values
print(X)
y = df1.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))
X_train.shape


```

## OUTPUT:
### Dataset
![s1](https://user-images.githubusercontent.com/113674204/191887944-59fbffdc-6651-42fb-92c3-00c6b2e0ed94.png)
### Checking for Null Values
![s2](https://user-images.githubusercontent.com/113674204/191889421-130b4434-abbf-4bfe-ba72-76ee78b0e0ec.png)
### Checking for duplicate values
![s3](https://user-images.githubusercontent.com/113674204/191889603-321c821d-c712-4db9-a618-df744cfd8c93.png)
### Describing Data
![s4](https://user-images.githubusercontent.com/113674204/191889726-3416435f-f4a3-4b9b-beeb-0a73b110108d.png)
### Checking for outliers in Exited Column
![s5](https://user-images.githubusercontent.com/113674204/191889863-91e37e9f-506a-4fc2-ba22-8f79556bdf2e.png)
### Normalized Dataset
![s6](https://user-images.githubusercontent.com/113674204/191890023-8173556f-78f6-4ab3-b49d-f657019637f9.png)
### Describing Normalized Data
![s7](https://user-images.githubusercontent.com/113674204/191890114-4133f44f-aa4d-4a4a-9bf4-2b38780a7cd0.png)
### X - Values
![s8](https://user-images.githubusercontent.com/113674204/191890233-a820a7e2-950b-417c-9321-876cac1ad005.png)
### Y - Value
![s9](https://user-images.githubusercontent.com/113674204/191890310-dcddf70f-8a71-442d-8fcc-bb1713f87763.png)
### X_train values
![s10](https://user-images.githubusercontent.com/113674204/191890378-27bb1453-8fe3-42ba-9784-85a7d68eac78.png)
### X_train Size
![s11](https://user-images.githubusercontent.com/113674204/191890456-1f4c4f42-60b0-45dc-b706-804316a05680.png)
### X_test values
![s12](https://user-images.githubusercontent.com/113674204/191890578-670b2bfd-73cb-4845-9162-9a18802bb52e.png)
### X_test Size
![s13](https://user-images.githubusercontent.com/113674204/191890651-c9c264ae-eb6c-4ecf-a92c-cfcd16991614.png)
### X_train shape
![s14](https://user-images.githubusercontent.com/113674204/191890731-75551863-4e25-4f0d-ad7b-54b922f7c9c7.png)





## RESULT
Data preprocessing is performed in a data set downloaded from Kaggle.
