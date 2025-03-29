# 📌 Importing the Dataset in Machine Learning

## 📖 Introduction
To build a Machine Learning model, we need data. Importing a dataset is the first step in any ML project. In Python, we commonly use libraries like **Pandas**, **NumPy**, and **Scikit-learn** to load datasets from various sources.

## 🏗️ Sources of Datasets
Datasets can come from different sources, including:
- **CSV Files** (`.csv`)
- **Excel Files** (`.xlsx`)
- **Databases** (SQL, NoSQL)
- **Online Repositories** (Kaggle, UCI, Google Drive)
- **Built-in Datasets** (Scikit-learn, TensorFlow, etc.)

## 🚀 Importing a Dataset
Here’s how to import datasets from different sources:

### 1️⃣ Importing a CSV File
The most common file format for datasets is CSV (Comma-Separated Values).
```python
import pandas as pd

df = pd.read_csv("dataset.csv")  # Replace with actual file path
print(df.head())  # Display the first 5 rows
```

### 2️⃣ Importing an Excel File
If your dataset is stored in an Excel file, use `read_excel`.
```python
df = pd.read_excel("dataset.xlsx", sheet_name="Sheet1")
print(df.head())
```

### 3️⃣ Importing a Dataset from Scikit-learn
Scikit-learn provides built-in datasets like Iris and Boston Housing.
```python
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
```

### 4️⃣ Importing a Dataset from an Online Source (Kaggle, UCI, etc.)
If the dataset is hosted online, use `pd.read_csv` with a URL.
```python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
print(df.head())
```

### 5️⃣ Importing Data from a Database
For SQL databases, use `sqlite3` or `SQLAlchemy`.
```python
import sqlite3

conn = sqlite3.connect("database.db")
df = pd.read_sql_query("SELECT * FROM table_name", conn)
print(df.head())
```

## 🔍 Handling Missing Data
Sometimes datasets have missing values. We can handle them using Pandas.
```python
print(df.isnull().sum())  # Check missing values
df.fillna(df.mean(), inplace=True)  # Fill missing values with mean
```

## 🏁 Conclusion
Importing datasets is an essential step in ML projects. Choosing the right method depends on the data source. Once the dataset is imported, we can proceed with data preprocessing and model building. 🚀

