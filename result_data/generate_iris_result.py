import time
import pandas as pd
from model import XGBoostExplainer, generate_results
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
y[y == 2] = 0  # converte em binario

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=1)
model.fit(X_train, y_train)
y_pred = model.predict(X)

explainer = XGBoostExplainer(model, X_train)

print(len(X))
print(model.score(X_test, y_test))

start = time.time()
generate_results(explainer, X, y_pred, 0,
                 'result_data/results/iris/', reorder="asc")
end = time.time()
length = end - start
print("time class 0", length, "sec")

start = time.time()
generate_results(explainer, X, y_pred, 1,
                 'result_data/results/iris/', reorder="asc")
end = time.time()
length = end - start
print("time class 1", length, "sec")