import time
import pandas as pd
from model import XGBoostExplainer, generate_results
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pmlb import fetch_data

mushroom = fetch_data('mushroom')
mushroom.head()

X = mushroom.drop('target', axis=1)
y = mushroom['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

model = XGBClassifier(n_estimators=30, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

explainer = XGBoostExplainer(model, X_train)

print(len(X))
print(len(X_test))
print(model.score(X_test, y_test))

X_class_0 = X_test[y_pred == 0].sample(n=100, random_state=42)
X_class_1 = X_test[y_pred == 1].sample(n=100, random_state=42)

y_pred_series = pd.Series(y_pred, index=X_test.index)
y_class_0 = y_pred_series.loc[X_class_0.index]
y_class_1 = y_pred_series.loc[X_class_1.index]

start = time.time()
generate_results(explainer, X_class_0, y_class_0, 0,
                 'result_data/results/mushroom/', reorder="asc")
end = time.time()
length = end - start
print("time class 0", length, "sec")

start = time.time()
generate_results(explainer, X_class_1, y_class_1, 1,
                 'result_data/results/mushroom/', reorder="asc")
end = time.time()
length = end - start
print("time class 1", length, "sec")