import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load the dataset (with headers)
df = pd.read_csv("train.csv")

# Combine title and description into a single text column
df['label'] = df['Class Index'].astype(int)
df['text'] = df['Title'] + " " + df['Description']

# Features and labels
X = df['text']
y = df['label'] - 1  # Convert labels from 1-4 to 0-3

# Split the dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression w/ Bag of Words
bow_vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_bow, y_train)
y_pred_lr = lr_model.predict(X_test_bow)

print("\nLogistic Regression (Bag of Words):")
print(classification_report(y_test, y_pred_lr))

# SVM w/ TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)

print("\nSupport Vector Machine (TF-IDF):")
print(classification_report(y_test, y_pred_svm))
