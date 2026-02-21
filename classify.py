import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK data for lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load the dataset
print("Loading dataset... this may take a moment on a MacBook Air.")
df = pd.read_csv('complaints_processed.csv')

# 2. Cleaning
df = df.dropna(subset=['narrative'])

# 3. Sampling for Research (40,000 rows)
df_sample = df.sample(40000, random_state=42)

# --- NEW: LEMMATIZATION SECTION ---
print("Applying Lemmatization (reducing words to root form)...")
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    # Converts 'paying', 'paid' -> 'pay'
    return " ".join([lemmatizer.lemmatize(word) for word in str(text).split()])

df_sample['narrative'] = df_sample['narrative'].apply(lemmatize_text)
# ----------------------------------

X = df_sample['narrative']
y = df_sample['product']

# 4. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorization (N-grams + Stop Words)
print("Vectorizing text data (N-grams: 1,2)...")
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Model Training & Comparison
print("\n--- Training Models for Comparative Analysis ---")

# Model A: Logistic Regression
print("Training Model A: Logistic Regression...")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Model B: Naive Bayes
print("Training Model B: Naive Bayes...")
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
y_pred_nb = model_nb.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Model C: Random Forest
print("Training Model C: Random Forest (Advanced)...")
model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
model_rf.fit(X_train_tfidf, y_train)
y_pred_rf = model_rf.predict(X_test_tfidf)
acc_rf = accuracy_score(y_test, y_pred_rf)

# 7. Print Comparative Results for Research Paper
print("\n" + "="*45)
print("FINAL RESEARCH COMPARISON METRICS")
print(f"Logistic Regression Accuracy: {acc_lr:.2%}")
print(f"Naive Bayes Accuracy:         {acc_nb:.2%}")
print(f"Random Forest Accuracy:       {acc_rf:.2%}")
print("="*45)

# 8. Detailed Statistical Report
print("\nDetailed Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# 9. Generate Confusion Matrix Visual
print("\nGenerating Confusion Matrix Visual...")
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model_lr.classes_, 
            yticklabels=model_lr.classes_)
plt.title('Research Figure 1: Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.tight_layout()
plt.savefig('research_confusion_matrix.png')

# 10. Save the best 'Brain' for the Web App
joblib.dump(model_lr, 'complaint_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n✅ PROCESS COMPLETE!")
print("1. 'complaint_model.pkl' updated with Lemmatized features.")
print("2. 'research_confusion_matrix.png' saved.")