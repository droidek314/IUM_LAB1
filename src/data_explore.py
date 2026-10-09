import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Reading .env
load_dotenv()
test_size = float(os.getenv("TEST_SIZE", 0.2))
random_state = int(os.getenv("RANDOM_STATE", 42))

# DataFrame
iris = load_iris(as_frame=True)
df = iris.frame

print(df.head())
print(df.shape)
print(df.describe())

# Data separation
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Model testing
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print(report)

with open("wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)