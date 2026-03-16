import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("../data/churn.csv")

model = GradientBoostingClassifier()

model.fit(X_train, y_train)

os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/churn_model.pkl")

print("Churn prediction model trained")
