#!/usr/bin/env python3
"""
Focused training script that trains model specifically from combined_dataset.csv
Uses only the comprehensive combined dataset for clean, single-source training
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import os

def preprocess_text(text):
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("🎯 FOCUSED TRAINING FROM COMBINED DATASET")
print("=" * 60)
print("📁 Training model specifically from combined_dataset.csv")

# Load the combined dataset
combined_dataset_path = 'data/combined_dataset.csv'

if not os.path.exists(combined_dataset_path):
    print(f"❌ Combined dataset not found at {combined_dataset_path}")
    print("💡 Please run the comprehensive training script first to create combined_dataset.csv")
    exit(1)

try:
    df = pd.read_csv(combined_dataset_path)
    print(f"📊 Loaded combined dataset: {len(df)} records")
except Exception as e:
    print(f"❌ Error loading combined dataset: {e}")
    exit(1)

# Validate dataset structure
if 'text' not in df.columns or 'label' not in df.columns:
    print("❌ Combined dataset must have 'text' and 'label' columns")
    exit(1)

# Analyze dataset
print(f"\n📊 Dataset Analysis:")
safe_count = len(df[df['label'] == 0])
toxic_count = len(df[df['label'] == 1])
print(f"   🟢 Safe messages: {safe_count} ({safe_count/len(df)*100:.1f}%)")
print(f"   🔴 Toxic messages: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")

# Clean data
print(f"\n🧹 Cleaning data...")
df = df.dropna(subset=['text'])
df = df[df['text'].str.len() > 0]
print(f"   📝 After cleaning: {len(df)} records")

# Preprocess text data
print(f"🧹 Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove empty texts after preprocessing
df = df[df['processed_text'].str.len() > 0]
print(f"📝 After preprocessing: {len(df)} records")

# Split the data
X = df['processed_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📚 Training set: {len(X_train)} records")
print(f"🧪 Test set: {len(X_test)} records")

# Create and train the vectorizer
print(f"\n📊 Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=1,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"📈 Vectorizer created with {len(vectorizer.get_feature_names_out())} features")

# Train the model
print(f"\n🎯 Training model from combined dataset...")
model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    C=10.0,  # Optimal balance of confidence and generalization
    solver='liblinear',
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Evaluate the model
print(f"\n📈 Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Toxic']))

# Test comprehensive examples
print(f"\n🧪 TESTING COMPREHENSIVE EXAMPLES:")
test_phrases = [
    # Previously fixed issues
    "hey girl", "hey girl how are you", "hey girl you look great",
    "hey waste", "hey waste of time", "hey waste you are useless",
    "jerk", "you jerk", "you are a jerk", "stupid jerk",
    
    # Basic safe phrases
    "hello", "hi", "hey", "good morning", "how are you", "thank you", "please", "sorry",
    "good job", "great work", "nice to meet you", "have a good day", "take care",
    
    # Basic toxic phrases
    "fucking idiot", "damn you", "you are too black", "slut", "bitch", "whore",
    "asshole", "bastard", "moron", "idiot", "fool", "scumbag",
    
    # Edge cases
    "waste fellow", "waste of space", "useless fellow", "worthless fellow",
    "pathetic jerk", "disgusting jerk", "lousy jerk", "miserable jerk",
    
    # Mixed phrases
    "hello you jerk", "hey girl you are amazing", "good morning asshole",
    "thank you bastard", "nice to meet you slut", "how are you moron",
    
    # Complex sentences
    "you are a complete waste of human resources and i hate you",
    "hey girl you are the most amazing person i have ever met",
    "what a stupid jerk you are completely useless and pathetic",
    "good morning everyone hope you have a wonderful day",
]

for phrase in test_phrases:
    processed = preprocess_text(phrase)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0])
    label = "Safe Message" if prediction == 0 else "Toxic Content Detected"
    
    # Determine expected classification
    toxic_words = ['jerk', 'waste', 'idiot', 'moron', 'fool', 'asshole', 'bastard', 
                   'scumbag', 'slut', 'bitch', 'whore', 'fucking', 'damn']
    
    is_toxic = any(word in phrase.lower() for word in toxic_words)
    expected = "Toxic" if is_toxic else "Safe"
    status = "✅" if (prediction == 1 and is_toxic) or (prediction == 0 and not is_toxic) else "❌"
    
    print(f"   {status} '{phrase}' → {label} ({confidence:.2%} confidence)")

# Ensure models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"📁 Created {models_dir} directory")

# Save the model and vectorizer
print(f"\n💾 Saving focused model and vectorizer...")
joblib.dump(model, 'models/toxic_classifier.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print(f"✅ FOCUSED TRAINING COMPLETE!")
print(f"📊 Training Summary:")
print(f"   📁 Dataset: combined_dataset.csv ({len(df)} records)")
print(f"   🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"   💾 Model: saved to models/ directory")
print(f"💡 Restart Flask app to use focused model")
