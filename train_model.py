import os
os.makedirs("model", exist_ok=True)
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("dataset/emails.csv")

# 2. Separate input and output
X = df["text"]
y = df["label"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

# 7. Save trained model
joblib.dump(model, "model/spam_model.pkl")

print("Model saved successfully!")