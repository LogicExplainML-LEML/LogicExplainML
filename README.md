# Logic Explain Machine Learning - LEML

`logic-explain-ml` is a Python library for explaining predictions of Machine Learning models.
It provides an easy-to-use interface to inspect feature importance and generate human-readable explanations for individual predictions.
By this date it only explains XGBoost models.

## Installation

You can install the package via pip:

```bash
pip install logic-explain-ml
````

## Features

* Fit an explainer for XGBoost models.
* Compute feature importance.
* Generate interpretable explanations for individual predictions.

## Usage

Here's a simple example using the Iris dataset and `XGBClassifier`:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

from logic_explain_ml.xgboost import XGBoostExplainer

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Convert to binary classification
y[y == 2] = 0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Train an XGBoost classifier
xgbc = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
xgbc.fit(X_train, y_train)

# Make predictions
preds = xgbc.predict(X_test)
print(preds)

# Initialize and fit the explainer
explainer = XGBoostExplainer(xgbc, X)
explainer.fit()

# Inspect feature importance
print(xgbc.feature_importances_, xgbc.feature_names_in_)

# Explain a single sample
sample = [5.5, 4.2, 1.4, 0.2]
exp = explainer.explain(sample, reorder="asc")
print(exp)
```

### Example Output

```text
init: -0.73
prob: -2.96
tree prob: 4.65
[petal length (cm) == 1.4]
```

## License

This project is licensed under the MIT License.



