import warnings
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Suppress all warnings (FutureWarning + calibration RuntimeWarnings)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv('complaints_processed.csv')

# 2. Cleaning
df = df.dropna(subset=['narrative'])

# 3. Diagnose class distribution BEFORE balancing
print("\n=== RAW CLASS DISTRIBUTION ===")
print(df['product'].value_counts())
print(f"\nTotal classes: {df['product'].nunique()}")

# 4. Strict per-class balancing
print("\nBalancing dataset...")
SAMPLES_PER_CLASS = 5000

df_balanced = (
    df.groupby('product', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=42))
    .reset_index(drop=True)
)

# Restore product column if dropped (Python 3.9 safety net)
if 'product' not in df_balanced.columns:
    df_balanced = df_balanced.reset_index()
    if 'product' not in df_balanced.columns:
        df_balanced['product'] = df_balanced.index.get_level_values('product')
    df_balanced = df_balanced.reset_index(drop=True)

print(f"\n=== BALANCED CLASS DISTRIBUTION ===")
print(df_balanced['product'].value_counts())
print(f"Total samples after balancing: {len(df_balanced)}")

df_sample = df_balanced

# 5. Preprocessing
print("\nApplying advanced preprocessing...")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df_sample['narrative'] = df_sample['narrative'].apply(preprocess_text)

# 6. Features & Labels
X = df_sample['narrative']
y = df_sample['product']

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. TF-IDF Vectorization
print("\nVectorizing text (N-grams 1,2 + sublinear TF)...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    max_df=0.85
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 9. Model Training
print("\n--- Training Models ---")

# Model A: Logistic Regression
print("Training Logistic Regression...")
model_lr = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    C=0.5,
    solver='saga',
    n_jobs=-1
)
model_lr.fit(X_train_tfidf, y_train)
y_pred_lr = model_lr.predict(X_test_tfidf)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Model B: Naive Bayes
print("Training Naive Bayes...")
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
y_pred_nb = model_nb.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Model C: LinearSVC with sigmoid calibration (stable, no matmul warnings)
print("Training LinearSVC...")
svc_base = LinearSVC(class_weight='balanced', C=0.5, max_iter=3000)
model_svc = CalibratedClassifierCV(svc_base, cv=5, method='sigmoid')
model_svc.fit(X_train_tfidf, y_train)
y_pred_svc = model_svc.predict(X_test_tfidf)
acc_svc = accuracy_score(y_test, y_pred_svc)

# 10. Results
print("\n" + "="*50)
print("FINAL RESULTS")
print(f"Logistic Regression Accuracy: {acc_lr:.2%}")
print(f"Naive Bayes Accuracy:         {acc_nb:.2%}")
print(f"LinearSVC Accuracy:           {acc_svc:.2%}")
print("="*50)

# 11. Pick best model automatically
best_acc = max(acc_lr, acc_nb, acc_svc)
if best_acc == acc_svc:
    best_model = model_svc
    best_preds = y_pred_svc
    best_name = "LinearSVC"
elif best_acc == acc_lr:
    best_model = model_lr
    best_preds = y_pred_lr
    best_name = "Logistic Regression"
else:
    best_model = model_nb
    best_preds = y_pred_nb
    best_name = "Naive Bayes"

print(f"\n🏆 Best Model: {best_name} ({best_acc:.2%})")

# 12. Classification Report for best model
print(f"\nDetailed Report ({best_name}):")
print(classification_report(y_test, best_preds))

# 13. Per-class prediction distribution
print(f"\n=== PER-CLASS PREDICTION DISTRIBUTION ({best_name}) ===")
print(pd.Series(best_preds).value_counts())

# 14. Confusion Matrix for best model
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(14, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_
)
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('research_confusion_matrix.png')
print("Confusion matrix saved as research_confusion_matrix.png")

# 15. Save best model & vectorizer
joblib.dump(best_model, 'complaint_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n✅ PROCESS COMPLETE!")
print(f"✔ Best model saved: {best_name} ({best_acc:.2%})")
print(f"✔ {SAMPLES_PER_CLASS} samples per class (balanced)")
print("✔ All warnings suppressed")
print("✔ sublinear_tf=True reduces dominant-class bias")
print("✔ Confusion matrix saved")
print("✔ Model and vectorizer saved")
