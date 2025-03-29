# 📌 Splitting the Dataset into the Training Set and Test Set

## 📖 Introduction
In Machine Learning, we split the dataset into **training** and **test** sets to evaluate model performance. This prevents overfitting and ensures the model generalizes well to new data.

## 🏗️ Why Split the Dataset?
- **Training Set**: Used to train the model
- **Test Set**: Used to evaluate model performance on unseen data
- **Prevents overfitting**: Ensures the model does not just memorize the training data

## 🚀 Splitting the Dataset Using Scikit-learn
We use `train_test_split` from Scikit-learn to split the dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")

# Define features (X) and target (y)
X = df.drop("target_column", axis=1)
y = df["target_column"]

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display dataset shapes
print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)
```

## 🔍 Parameters in `train_test_split`
- `test_size=0.2`: 20% of the data is used for testing (adjustable)
- `random_state=42`: Ensures reproducibility (use any fixed number)
- `shuffle=True`: Shuffles data before splitting (default is `True`)

## 🎯 Alternative: Stratified Splitting (For Imbalanced Data)
If the dataset has imbalanced classes, we use **stratified splitting**.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

## 🏁 Conclusion
Splitting the dataset is a crucial step to assess model performance. A proper split ensures that the model learns well without overfitting or underfitting. 🚀