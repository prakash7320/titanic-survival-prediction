import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Remove unwanted columns
df.drop(["PassengerId","Name","Ticket","Cabin"], axis=1, inplace=True)

# Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical columns
encoder = {}

for col in ["Sex","Embarked"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoder[col] = le

# Features & Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))

# Save Model
joblib.dump(model,"model.pkl")
joblib.dump(encoder,"encoder.pkl")

print("Model Saved Successfully")