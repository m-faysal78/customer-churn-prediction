import joblib
from sklearn.metrics import accuracy_score
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("../data/churn.csv")

model = joblib.load("../model/churn_model.pkl")

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
