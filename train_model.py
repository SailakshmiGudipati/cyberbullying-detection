#!/usr/bin/env python3
"""
Complete ML training script for toxic comment detection
Handles dataset creation, splitting, and model training in one script
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
import glob

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

print("🎯 COMPLETE ML TRAINING PIPELINE")
print("=" * 60)
print("📁 Handling dataset creation, splitting, and model training")

# Check if we need to create combined dataset
data_dir = 'data'
dataset_files = [
    'dataset.csv',
    'old_dataset.csv', 
    'previous_dataset.csv',
    'racist_fixed_dataset.csv',
    'test.csv',
    'train.csv',
    'val.csv',
    'variations_dataset.csv'
]

# Check if individual datasets exist
individual_datasets = []
for dataset_file in dataset_files:
    file_path = os.path.join(data_dir, dataset_file)
    if os.path.exists(file_path):
        individual_datasets.append(file_path)

combined_dataset_path = os.path.join(data_dir, 'combined_dataset.csv')

if individual_datasets and not os.path.exists(combined_dataset_path):
    print("\n🔄 Creating combined dataset from individual files...")
    
    # Load all individual datasets
    all_data = []
    total_records = 0
    
    for dataset_file in dataset_files:
        file_path = os.path.join(data_dir, dataset_file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"📊 Loaded {dataset_file}: {len(df)} records")
                
                # Ensure consistent column names
                if 'text' not in df.columns or 'label' not in df.columns:
                    # Try to identify text and label columns
                    text_col = None
                    label_col = None
                    
                    for col in df.columns:
                        if col.lower() in ['text', 'message', 'comment', 'content', 'sentence']:
                            text_col = col
                        elif col.lower() in ['label', 'toxic', 'class', 'target', 'classification']:
                            label_col = col
                    
                    if text_col and label_col:
                        df = df.rename(columns={text_col: 'text', label_col: 'label'})
                        print(f"   🔄 Renamed columns: {text_col}→text, {label_col}→label")
                    else:
                        print(f"   ⚠️  Skipping {dataset_file}: couldn't identify text/label columns")
                        continue
                
                # Keep only text and label columns
                df = df[['text', 'label']].copy()
                all_data.append(df)
                total_records += len(df)
                
            except Exception as e:
                print(f"   ❌ Error loading {dataset_file}: {e}")
    
    if not all_data:
        print("❌ No valid datasets found!")
        exit(1)
    
    # Combine all datasets
    print(f"\n📈 Combining {total_records} total records...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    print(f"🔄 Removing duplicates...")
    combined_df = combined_df.drop_duplicates(subset=['text'])
    print(f"   📝 After removing duplicates: {len(combined_df)} records")
    
    # Clean data
    print(f"🧹 Cleaning data...")
    combined_df = combined_df.dropna(subset=['text'])
    combined_df = combined_df[combined_df['text'].str.len() > 0]
    print(f"   📝 After cleaning: {len(combined_df)} records")
    
    # Save combined dataset
    combined_df.to_csv(combined_dataset_path, index=False)
    print(f"💾 Saved combined dataset: {combined_dataset_path}")
    
else:
    print("\n📁 Using existing combined dataset...")

# Load combined dataset
if not os.path.exists(combined_dataset_path):
    print(f"❌ Combined dataset not found at {combined_dataset_path}")
    print("💡 Run script with individual datasets to create combined dataset")
    exit(1)

try:
    df = pd.read_csv(combined_dataset_path)
    print(f"📊 Loaded combined dataset: {len(df)} records")
except Exception as e:
    print(f"❌ Error loading combined dataset: {e}")
    exit(1)

# Analyze dataset
print(f"\n📊 Dataset Analysis:")
safe_count = len(df[df['label'] == 0])
toxic_count = len(df[df['label'] == 1])
print(f"   🟢 Safe messages: {safe_count} ({safe_count/len(df)*100:.1f}%)")
print(f"   🔴 Toxic messages: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")

# Create train/val/test splits
print(f"\n🔄 Creating train/val/test splits...")

X = df['text']
y = df['label']

# First split: 80% train, 20% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: Split temp into 50% val, 50% test (10% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"📊 Final Split:")
print(f"   📚 Training: {len(X_train)} records ({len(X_train)/len(df)*100:.1f}%)")
print(f"   🧪 Validation: {len(X_val)} records ({len(X_val)/len(df)*100:.1f}%)")
print(f"   🧪 Test: {len(X_test)} records ({len(X_test)/len(df)*100:.1f}%)")

# Save splits
train_df = pd.DataFrame({'text': X_train, 'label': y_train})
val_df = pd.DataFrame({'text': X_val, 'label': y_val})
test_df = pd.DataFrame({'text': X_test, 'label': y_test})

train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

print(f"💾 Saved dataset splits:")
print(f"   📁 train.csv: {len(train_df)} records")
print(f"   📁 val.csv: {len(val_df)} records")
print(f"   📁 test.csv: {len(test_df)} records")

# Preprocess text data
print(f"\n🧹 Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove empty texts after preprocessing
df = df[df['processed_text'].str.len() > 0]
print(f"📝 After preprocessing: {len(df)} records")

# Split the data for training
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
print(f"\n🎯 Training final model...")
model = LogisticRegression(
    random_state=42,
    max_iter=2000,
    C=10.0,
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
    # Basic safe phrases
    "hello", "hi", "hey", "good morning", "how are you", "thank you", "please", "sorry",
    "good job", "great work", "nice to meet you", "have a good day", "take care",
    
    # Basic toxic phrases
    "fucking idiot", "damn you", "you are too black", "slut", "bitch", "whore",
    "asshole", "bastard", "moron", "idiot", "fool", "scumbag",
    
    # Previously fixed issues
    "hey girl", "hey girl how are you", "hey girl you look great",
    "hey waste", "hey waste of time", "hey waste you are useless",
    "jerk", "you jerk", "you are a jerk", "stupid jerk",
    
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
print(f"\n💾 Saving final model and vectorizer...")
joblib.dump(model, os.path.join(models_dir, 'toxic_classifier.pkl'))
joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))

print(f"✅ COMPLETE ML TRAINING PIPELINE FINISHED!")
print(f"🎯 Summary:")
print(f"   📁 Dataset: {len(df)} records processed")
print(f"   📊 Splits: train({len(X_train)})/val({len(X_val)})/test({len(X_test)})")
print(f"   🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"   💾 Model: saved to {models_dir}/")
print(f"🚀 Ready for production deployment!")
