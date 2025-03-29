# 📌 Importing the Libraries in Machine Learning

## 📖 Introduction
When working on a Machine Learning project, the first step is to import the necessary libraries. These libraries provide essential functions for data manipulation, visualization, model building, and evaluation.

## 🏗️ Purpose of Importing Libraries
Each library in Python serves a specific purpose in ML projects:
- **NumPy**: Supports numerical computations.
- **Pandas**: Helps in data manipulation and analysis.
- **Matplotlib & Seaborn**: Used for data visualization.
- **Scikit-learn**: Provides ML algorithms and utilities.
- **TensorFlow & PyTorch**: Deep learning frameworks.

## 🚀 Importing Essential Libraries
Below is an example of how to import common ML libraries:

```python
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualizations
from sklearn.model_selection import train_test_split  # For dataset splitting
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LinearRegression  # Example ML model
import tensorflow as tf  # Deep learning framework
import torch  # Alternative deep learning framework
```

## 🔍 Explanation of Each Library
### 1️⃣ NumPy (`import numpy as np`)
Used for handling arrays and performing mathematical operations efficiently.
#### 🔹 Example:
```python
arr = np.array([1, 2, 3])
print(arr * 2)  # Output: [2 4 6]
```

### 2️⃣ Pandas (`import pandas as pd`)
Used for data manipulation, including reading and modifying datasets.
#### 🔹 Example:
```python
df = pd.DataFrame({'Name': ['Dokja', 'Jeha'], 'Score': [90, 85]})
print(df)
```

### 3️⃣ Matplotlib & Seaborn (`import matplotlib.pyplot as plt`, `import seaborn as sns`)
Used for data visualization.
#### 🔹 Example:
```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

### 4️⃣ Scikit-learn (`from sklearn...`)
Provides ML utilities, such as data splitting and model training.
#### 🔹 Example:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 5️⃣ TensorFlow & PyTorch (`import tensorflow as tf`, `import torch`)
Used for deep learning model development.
#### 🔹 Example (TensorFlow):
```python
model = tf.keras.models.Sequential()
```
#### 🔹 Example (PyTorch):
```python
tensor = torch.tensor([1, 2, 3])
```

## 🏁 Conclusion
Importing the right libraries is crucial for a smooth ML workflow. Understanding their roles helps in efficient model development and data processing. 🚀

