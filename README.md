# Customer Churn Prediction

Machine learning project that predicts whether a customer will churn based on behavioral and demographic features.

## Problem Statement

Customer churn is a major challenge for subscription-based businesses. Predicting churn allows organizations to proactively retain customers through targeted interventions.

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
Gradient Boosting  

## Project Structure

customer-churn-prediction

data/
churn.csv

src/
preprocess.py
train.py
evaluate.py

requirements.txt
README.md


## Workflow

1. Data preprocessing and feature engineering
2. Feature scaling using StandardScaler
3. Train/test split
4. Model training using Gradient Boosting
5. Model evaluation

## Model Used

Gradient Boosting Classifier

Benefits:
- Strong performance on structured datasets
- Handles complex feature interactions
- High predictive accuracy

## Evaluation Metric

Accuracy Score

## How to Run

Install dependencies

pip install -r requirements.txt

Train the model

python src/train.py

Evaluate the model

python src/evaluate.py

## Example Output

Model Accuracy: 0.86

## Future Improvements

Add feature importance analysis

Integrate model monitoring

Deploy model as an API using FastAPI

Add dashboard visualization
