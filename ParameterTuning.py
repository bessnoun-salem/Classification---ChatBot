# 📦 Import required libraries
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# 🧹 Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatization
    return ' '.join(words)

# 📥 Load the dataset (must contain 'response_text' and 'class' columns)
data = pd.read_csv("Sheet_1.csv", usecols=["response_text", "class"], encoding='latin-1')
data = data.dropna()  # Remove missing values

# 🧽 Clean the text column
data["clean_text"] = data["response_text"].apply(clean_text)

# 🎯 Define features and labels
X = data["clean_text"]
y = data["class"]

# 🔢 Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X_vec = vectorizer.fit_transform(X)

# 🔀 Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 🤖 1. Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# 🌳 2. Train a Decision Tree classifier
dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 📊 Evaluate both models
print("===== Naive Bayes Classifier =====")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))

print("\n===== Decision Tree Classifier =====")
print("Accuracy:", accuracy_score(y_test, dt_preds))
print(classification_report(y_test, dt_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_preds))
