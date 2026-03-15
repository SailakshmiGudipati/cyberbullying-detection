import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DatasetPreparer:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove punctuation and special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_offensive_words_dataset(self):
        """Create dataset using offensive words list"""
        print("Creating offensive words dataset...")
        
        # Comprehensive list of offensive words
        offensive_words = [
            'ass', 'bastard', 'bitch', 'crap', 'damn', 'fuck', 'hell', 'shit', 'whore',
            'slut', 'dick', 'pussy', 'cock', 'cunt', 'motherfucker', 'son of a bitch',
            'asshole', 'douchebag', 'faggot', 'nigger', 'nigga', 'retard', 'idiot',
            'moron', 'stupid', 'dumb', 'loser', 'jerk', 'punk', 'wanker', 'twat',
            'kill yourself', 'go die', 'rot in hell', 'burn in hell'
        ]
        
        # Create offensive sentences
        offensive_sentences = []
        for word in offensive_words:
            offensive_sentences.extend([
                f"You are a {word}",
                f"{word} you",
                f"Go {word} yourself",
                f"Stupid {word}",
                f"Damn {word}"
            ])
        
        # Create non-offensive sentences
        positive_words = [
            'love', 'happy', 'great', 'awesome', 'wonderful', 'amazing', 'excellent',
            'fantastic', 'beautiful', 'kind', 'helpful', 'friend', 'family', 'success',
            'achievement', 'progress', 'hope', 'peace', 'joy', 'smile', 'laugh'
        ]
        
        non_offensive_sentences = []
        for word in positive_words:
            non_offensive_sentences.extend([
                f"You are {word}",
                f"Have a {word} day",
                f"That's very {word}",
                f"Feeling {word} today",
                f"{word} to see you"
            ])
        
        # Create DataFrame
        offensive_data = pd.DataFrame({
            'text': offensive_sentences,
            'label': [1] * len(offensive_sentences)
        })
        
        non_offensive_data = pd.DataFrame({
            'text': non_offensive_sentences,
            'label': [0] * len(non_offensive_sentences)
        })
        
        # Combine and shuffle
        combined_df = pd.concat([offensive_data, non_offensive_data], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created {len(combined_df)} samples using offensive words")
        return combined_df
    
    def create_twitter_style_dataset(self):
        """Create Twitter-style toxic comments dataset"""
        print("Creating Twitter-style dataset...")
        
        # Toxic Twitter-style comments
        toxic_comments = [
            "you're so stupid lol",
            "kill yourself already",
            "nobody likes you",
            "go die in a hole",
            "you're worthless trash",
            "stupid idiot",
            "dumbass motherfucker",
            "fuck you",
            "you're a piece of shit",
            "go to hell",
            "nobody cares about you",
            "you're pathetic",
            "loser",
            "freak",
            "weirdo",
            "get a life",
            "you're disgusting",
            "hate you",
            "die",
            "burn in hell",
            "go kill yourself"
        ]
        
        # Non-toxic Twitter-style comments
        non_toxic_comments = [
            "have a great day!",
            "love this so much",
            "amazing content",
            "thank you for sharing",
            "this is awesome",
            "so happy for you",
            "congratulations!",
            "beautiful",
            "inspiring",
            "wonderful",
            "perfect",
            "excellent work",
            "so proud",
            "keep it up",
            "you got this",
            "blessed",
            "grateful",
            "positive vibes",
            "smile more",
            "stay strong"
        ]
        
        # Create variations by adding common Twitter elements
        def add_twitter_variations(comments):
            variations = []
            prefixes = ['', 'omg ', 'lol ', 'wow ', 'smh ', 'tbh ', 'ngl ']
            suffixes = ['', ' lol', ' smh', ' tbh', ' fr', ' rn']
            
            for comment in comments:
                for prefix in prefixes:
                    for suffix in suffixes:
                        variations.append(prefix + comment + suffix)
            return variations
        
        toxic_variations = add_twitter_variations(toxic_comments)
        non_toxic_variations = add_twitter_variations(non_toxic_comments)
        
        # Limit to reasonable size
        toxic_variations = toxic_variations[:500]
        non_toxic_variations = non_toxic_variations[:500]
        
        # Create DataFrame
        toxic_df = pd.DataFrame({
            'text': toxic_variations,
            'label': [1] * len(toxic_variations)
        })
        
        non_toxic_df = pd.DataFrame({
            'text': non_toxic_variations,
            'label': [0] * len(non_toxic_variations)
        })
        
        # Combine and shuffle
        combined_df = pd.concat([toxic_df, non_toxic_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created {len(combined_df)} Twitter-style samples")
        return combined_df
    
    def create_realistic_comments_dataset(self):
        """Create more realistic comments dataset"""
        print("Creating realistic comments dataset...")
        
        # Realistic toxic comments
        realistic_toxic = [
            "I can't believe how stupid this is",
            "This is the worst thing I've ever seen",
            "You have no idea what you're talking about",
            "This makes me sick",
            "What a complete waste of time",
            "I hope this fails miserably",
            "Nobody in their right mind would think this is good",
            "This is absolutely terrible",
            "I'm disgusted by this",
            "What were you thinking?",
            "This is pathetic",
            "Complete garbage",
            "I regret wasting my time on this",
            "This is why nobody takes you seriously",
            "Absolutely awful",
            "I can't stand this",
            "This is insulting",
            "What a joke",
            "I'm so disappointed",
            "This should be deleted"
        ]
        
        # Realistic non-toxic comments
        realistic_non_toxic = [
            "I really enjoyed this",
            "Great work on this project",
            "This is very helpful",
            "Thank you for sharing this",
            "I learned a lot from this",
            "This is exactly what I was looking for",
            "Excellent explanation",
            "Very well done",
            "I appreciate this content",
            "This made my day",
            "Fantastic job",
            "Keep up the great work",
            "This is so useful",
            "I'm impressed with the quality",
            "Wonderful presentation",
            "This exceeded my expectations",
            "I'm grateful for this",
            "This is brilliant",
            "Outstanding work",
            "This is exactly what we needed"
        ]
        
        # Create DataFrame
        toxic_df = pd.DataFrame({
            'text': realistic_toxic * 10,  # Repeat to get more samples
            'label': [1] * len(realistic_toxic * 10)
        })
        
        non_toxic_df = pd.DataFrame({
            'text': realistic_non_toxic * 10,  # Repeat to get more samples
            'label': [0] * len(realistic_non_toxic * 10)
        })
        
        # Combine and shuffle
        combined_df = pd.concat([toxic_df, non_toxic_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created {len(combined_df)} realistic comment samples")
        return combined_df
    
    def combine_datasets(self, offensive_df=None, twitter_df=None, realistic_df=None):
        """Combine all datasets into one unified dataset"""
        print("Combining all datasets...")
        
        all_dfs = []
        
        if offensive_df is not None:
            all_dfs.append(offensive_df)
            print(f"Added {len(offensive_df)} samples from offensive words dataset")
        
        if twitter_df is not None:
            all_dfs.append(twitter_df)
            print(f"Added {len(twitter_df)} samples from Twitter-style dataset")
        
        if realistic_df is not None:
            all_dfs.append(realistic_df)
            print(f"Added {len(realistic_df)} samples from realistic comments dataset")
        
        if not all_dfs:
            print("No datasets to combine")
            return None
        
        # Combine all datasets
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean text for all samples
        combined_df['text'] = combined_df['text'].apply(self.clean_text)
        
        # Remove empty texts
        combined_df = combined_df[combined_df['text'].str.len() > 0]
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Final combined dataset: {len(combined_df)} samples")
        print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
        
        return combined_df
    
    def split_and_save(self, df, test_size=0.2, val_size=0.1):
        """Split dataset and save to files"""
        print("Splitting and saving dataset...")
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=42, stratify=train_val_df['label']
        )
        
        # Save datasets
        train_df.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.data_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.data_dir, 'test.csv'), index=False)
        
        # Save combined dataset
        df.to_csv(os.path.join(self.data_dir, 'dataset.csv'), index=False)
        
        print(f"Saved datasets:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Total: {len(df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_all_datasets(self):
        """Main method to prepare all datasets"""
        print("Starting dataset preparation...")
        
        # Create offensive words dataset
        offensive_df = self.create_offensive_words_dataset()
        
        # Create Twitter-style dataset
        twitter_df = self.create_twitter_style_dataset()
        
        # Create realistic comments dataset
        realistic_df = self.create_realistic_comments_dataset()
        
        # Combine all datasets
        combined_df = self.combine_datasets(offensive_df, twitter_df, realistic_df)
        
        if combined_df is not None:
            # Split and save
            train_df, val_df, test_df = self.split_and_save(combined_df)
            
            print("Dataset preparation completed successfully!")
            return train_df, val_df, test_df
        else:
            print("Failed to prepare datasets")
            return None, None, None

if __name__ == "__main__":
    # Create dataset preparer and run
    preparer = DatasetPreparer()
    train_df, val_df, test_df = preparer.prepare_all_datasets()
    
    if train_df is not None:
        print("\nDataset preparation completed!")
        print(f"Training set shape: {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print(f"Training set label distribution:\n{train_df['label'].value_counts()}")
