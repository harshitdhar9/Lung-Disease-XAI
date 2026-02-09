"""
Importing Libraries
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

"""
Loading Data
"""

data=pd.read_csv('cancer patient data sets.csv')

"""
Preprocessing Data
"""

data.columns=(
    data.columns.str.strip().str.lower().str.replace(' ','_')
)

print(data.head())
print(data.info())
print(data.shape)
print(data.describe())

data.drop(["index", "patient_id"], axis=1, inplace=True)
print(data["level"].unique())

mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
data["level"] = data["level"].map(mapping)

#print(data.info())

"""
Splitting Data
"""

X=data.drop(columns=["level"])
y=data["level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

"""
Model Implementation
"""

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

"""
Converting back from numpy array to Dataframe for SHAP
"""

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

model=LogisticRegression(class_weight='balanced',max_iter=500,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred),end="\n\n")
print(classification_report(y_test,y_pred),end="\n\n")
print(confusion_matrix(y_test,y_pred))

"""
Cross Validation Scores and Correlation
"""
scores = cross_val_score(model, X_train, y_train, cv=5)

print(scores)
print(scores.mean())

print(data.corr()["level"].sort_values(ascending=False))

"""
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
shap.plots.waterfall(shap_values[0, :, 0])
"""

import skfuzzy as fuzz

x_prob = np.arange(0, 1.01, 0.01)

low_membership = fuzz.trimf(x_prob, [0, 0, 0.5])
medium_membership = fuzz.trimf(x_prob, [0.2, 0.5, 0.8])
high_membership = fuzz.trimf(x_prob, [0.5, 1, 1])


def fuzzy_risk_interpretation(probabilities):
    """
    Convert model probability output into fuzzy linguistic explanation
    """

    labels = ["Low Risk", "Medium Risk", "High Risk"]

    interpretation = {}

    for i, prob in enumerate(probabilities):

        low_val = fuzz.interp_membership(x_prob, low_membership, prob)
        med_val = fuzz.interp_membership(x_prob, medium_membership, prob)
        high_val = fuzz.interp_membership(x_prob, high_membership, prob)

        interpretation[labels[i]] = {
            "Probability": round(prob, 3),
            "Low Membership": round(low_val, 3),
            "Medium Membership": round(med_val, 3),
            "High Membership": round(high_val, 3)
        }

    return interpretation

probs = model.predict_proba(X_test)
sample_prob = probs[0]

print("\nModel Probability Output:")
print(sample_prob)

print("\nFuzzy Risk Interpretation:")
fuzzy_output = fuzzy_risk_interpretation(sample_prob)

for risk_level, values in fuzzy_output.items():
    print(f"\n{risk_level}")
    for k, v in values.items():
        print(f"  {k}: {v}")
