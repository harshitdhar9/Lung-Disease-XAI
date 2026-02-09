import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("cancer patient data sets.csv")

data.columns = (
    data.columns.str.strip().str.lower().str.replace(' ','_')
)

data.drop(["index", "patient_id"], axis=1, inplace=True)

mapping = {"Low":0,"Medium":1,"High":2}
data["level"] = data["level"].map(mapping)

X = data.drop(columns=["level"])
y = data["level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(class_weight='balanced', max_iter=500)
model.fit(X_scaled,y)

joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns.tolist(),"columns.pkl")
