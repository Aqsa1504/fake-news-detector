import sys
import pickle
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path      = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model      = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))


news_text = sys.stdin.read().strip()

if not news_text:
    print("Error|0")
    sys.exit(1)


X           = vectorizer.transform([news_text])
prediction  = model.predict(X)[0]
probability = model.predict_proba(X).max()

result = "Fake" if prediction == 0 else "Real"

# Output format: Prediction|Confidence
print(f"{result}|{round(probability * 100, 2)}")