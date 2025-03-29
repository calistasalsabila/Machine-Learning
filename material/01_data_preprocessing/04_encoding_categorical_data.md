# 📌 Taking Care of Missing Data in Machine Learning

## 📖 Introduction
Missing data is a common problem in Machine Learning. Handling missing values properly is crucial for building accurate models. If left untreated, missing data can lead to biased results or errors during training.

## 🏗️ Common Causes of Missing Data
- **Human error**: Incomplete data entry
- **Sensor failure**: Hardware issues causing missing values
- **Data corruption**: Issues during data collection or storage
- **Filtering issues**: Data removed during preprocessing

## 🚀 Identifying Missing Data
Before handling missing values, we need to detect them.
```python
import pandas as pd

df = pd.read_csv("dataset.csv")  # Load dataset
print(df.isnull().sum())  # Count missing values per column
```

## 🔍 Methods to Handle Missing Data
### 1️⃣ Removing Missing Data
#### 🔹 Remove rows with missing values
```python
df_cleaned = df.dropna()
```
#### 🔹 Remove columns with many missing values
```python
df_cleaned = df.dropna(axis=1)
```
⚠️ **Use with caution!** This can lead to loss of valuable data.

### 2️⃣ Filling Missing Data
#### 🔹 Fill with a specific value
```python
df.fillna(0, inplace=True)
```
#### 🔹 Fill with the mean (for numerical data)
```python
df.fillna(df.mean(), inplace=True)
```
#### 🔹 Fill with the median (for skewed data)
```python
df.fillna(df.median(), inplace=True)
```
#### 🔹 Fill with the most frequent value (for categorical data)
```python
df.fillna(df.mode().iloc[0], inplace=True)
```

### 3️⃣ Using Interpolation
Interpolation estimates missing values based on existing data.
```python
df.interpolate(method='linear', inplace=True)
```

### 4️⃣ Using Machine Learning Models to Predict Missing Values
Sometimes, missing values can be predicted using ML models.
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[['column_name']] = imputer.fit_transform(df[['column_name']])
```

## 🏁 Conclusion
Handling missing data correctly ensures that ML models perform optimally. Choosing the right method depends on the dataset and the nature of missing values. 🚀