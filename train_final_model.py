import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class MentalHealthAbusiveTrainer:
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        
    def clean_text(self, text):
        """Clean text while preserving abusive word characteristics"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Keep some punctuation that might indicate abusive language
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\*\@\#\%]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_comprehensive_abusive_dataset(self):
        """Create comprehensive dataset with ALL categories including mental health"""
        print("Creating comprehensive abusive words dataset with mental health insults...")
        
        # Categories of abusive words
        
        # Sexual insults and prostitution terms
        sexual_insults = [
            'slut', 'whore', 'bitch', 'cunt', 'pussy', 'dick', 'cock', 'twat',
            'motherfucker', 'son of a bitch', 'bastard', 'asshole', 'douchebag',
            'wanker', 'prick', 'knob', 'bellend', 'minger', 'minge', 'prostitute',
            'hooker', 'call girl', 'escort', 'street walker', 'harlot', 'tramp',
            'tart', 'floozy', 'floozy', 'jezebel', 'hussy', 'strumpet'
        ]
        
        # Racial slurs
        racial_slurs = [
            'nigger', 'nigga', 'coon', 'spade', 'golliwog', 'paki', 'chink',
            'gook', 'kike', 'yid', 'raghead', 'towelhead', 'sandnigger'
        ]
        
        # Homophobic slurs
        homophobic_slurs = [
            'faggot', 'fag', 'poof', 'queer', 'dyke', 'lesbo', 'tranny',
            'shemale', 'ladyboy'
        ]
        
        # Disability insults
        disability_insults = [
            'retard', 'spastic', 'cripple', 'mong', 'idiot', 'moron', 'imbecile',
            'vegetable', 'windowlicker'
        ]
        
        # Mental health insults (NEW - as requested)
        mental_health_insults = [
            'mental', 'psycho', 'psychotic', 'schizo', 'schizophrenic', 'bipolar',
            'insane', 'crazy', 'mad', 'lunatic', 'nutcase', 'nuts', 'bonkers',
            'deranged', 'unstable', 'disturbed', 'sick', 'sicko', 'freak',
            'weirdo', 'mental patient', 'mental case', 'mental institution',
            'asylum', 'loony bin', 'madhouse', 'psych ward', 'straightjacket',
            'headcase', 'basket case', 'mental illness', 'psychopath', 'sociopath',
            'maniac', 'lunatic fringe', 'crazy person', 'madman', 'madwoman'
        ]
        
        # General insults
        general_insults = [
            'fuck', 'shit', 'crap', 'piss', 'bollocks', 'bugger', 'damn', 'hell',
            'arse', 'ass', 'bastard', 'git', 'pillock', 'plonker', 'wally',
            'dickhead', 'shithead', 'arsehole', 'asshole'
        ]
        
        # Body shaming insults
        body_shaming = [
            'fat', 'obese', 'chubby', 'overweight', 'plump', 'bloated', 'portly',
            'lardass', 'fatass', 'fatso', 'tubby', 'whale', 'pig', 'cow',
            'elephant', 'hippo', 'beached whale', 'ugly', 'hideous', 'disgusting',
            'gross', 'repulsive', 'unattractive', 'homely', 'plain', 'dog',
            'buttaface', 'butterface', 'skank', 'gross', 'fugly', 'troll',
            'gremlin', 'goblin', 'troll', 'ogre', 'beast'
        ]
        
        # Appearance shaming
        appearance_shaming = [
            'pimply', 'acne face', 'pizza face', 'zit face', 'ugly', 'hideous',
            'disfigured', 'deformed', 'mutant', 'freak', 'monster', 'creature',
            'gross', 'disgusting', 'revolting', 'repulsive', 'nasty', 'filthy',
            'dirty', 'smelly', 'stinky', 'rank', 'foul', 'vile'
        ]
        
        # Intelligence shaming
        intelligence_shaming = [
            'stupid', 'dumb', 'idiot', 'moron', 'retard', 'imbecile', 'cretin',
            'dimwit', 'nitwit', 'halfwit', 'simpleton', 'fool', 'jerk', 'dummy',
            'airhead', 'bimbo', 'ditz', 'dumbass', 'dumbfuck', 'numbskull',
            'peabrain', 'birdbrain', 'knucklehead', 'blockhead', 'thickhead'
        ]
        
        # Threatening language
        threats = [
            'kill yourself', 'go die', 'rot in hell', 'burn in hell', 'go kill yourself',
            'hang yourself', 'shoot yourself', 'stab yourself', 'jump off a bridge',
            'drink poison', 'end your life', 'commit suicide', 'disappear forever',
            'vanish from existence', 'i will kill you', 'i will find you', 'i will hurt you'
        ]
        
        # Combine all abusive categories
        all_abusive = (sexual_insults + racial_slurs + homophobic_slurs + 
                      disability_insults + mental_health_insults + general_insults + 
                      body_shaming + appearance_shaming + intelligence_shaming + threats)
        
        # Create variations for each abusive word/phrase
        abusive_samples = []
        
        for abusive in all_abusive:
            # Single word
            abusive_samples.append(abusive)
            
            # With punctuation
            abusive_samples.extend([
                f"{abusive}.",
                f"{abusive}!",
                f"{abusive}?",
                f"{abusive}!!",
                f"{abusive}???",
                f"{abusive}..."
            ])
            
            # With common prefixes
            prefixes = ['you', 'youre', 'you are', 'stupid', 'fucking', 'damn', 'fucking', 'ugly']
            for prefix in prefixes:
                abusive_samples.append(f"{prefix} {abusive}")
            
            # With common suffixes
            suffixes = ['you', 'yourself', 'your mom', 'your family', 'bitch', 'asshole', 'whore']
            for suffix in suffixes:
                abusive_samples.append(f"{abusive} {suffix}")
            
            # In sentences
            sentences = [
                f"go {abusive} yourself",
                f"what a {abusive}",
                f"you're such a {abusive}",
                f"{abusive}!",
                f"damn {abusive}",
                f"fucking {abusive}",
                f"stupid {abusive}",
                f"{abusive} bitch",
                f"you {abusive}",
                f"{abusive} you"
            ]
            abusive_samples.extend(sentences)
        
        # Create DataFrame
        abusive_df = pd.DataFrame({
            'text': abusive_samples,
            'label': [1] * len(abusive_samples)
        })
        
        print(f"Created {len(abusive_df)} abusive samples")
        return abusive_df
    
    def create_mental_health_dataset(self):
        """Create specific mental health shaming dataset"""
        print("Creating mental health shaming dataset...")
        
        # Mental health shaming phrases
        mental_health_phrases = [
            "you are mental",
            "you're psycho",
            "you're psychotic",
            "you're schizo",
            "you're schizophrenic",
            "you're bipolar",
            "you're insane",
            "you're crazy",
            "you're mad",
            "you're a lunatic",
            "you're a nutcase",
            "you're nuts",
            "you're bonkers",
            "you're deranged",
            "you're unstable",
            "you're disturbed",
            "you're sick",
            "you're a sicko",
            "you're a freak",
            "you're a weirdo",
            "you're a mental patient",
            "you're a mental case",
            "you belong in a mental institution",
            "you belong in an asylum",
            "you belong in a loony bin",
            "you belong in a madhouse",
            "you belong in a psych ward",
            "you need a straightjacket",
            "you're a headcase",
            "you're a basket case",
            "you have mental illness",
            "you're a psychopath",
            "you're a sociopath",
            "you're a maniac",
            "you're from the lunatic fringe",
            "you're a crazy person",
            "you're a madman",
            "you're a madwoman",
            "go to a mental hospital",
            "get mental help",
            "you need therapy",
            "you're mentally ill",
            "you have mental problems",
            "you're mentally unstable",
            "you're mentally disturbed",
            "you're mentally deranged",
            "you're psycho crazy",
            "you're insane mental"
        ]
        
        # Create variations
        mental_health_samples = []
        for phrase in mental_health_phrases:
            mental_health_samples.extend([
                phrase,
                f"{phrase}!",
                f"{phrase}??",
                f"what a {phrase}",
                f"such {phrase}",
                f"so {phrase}",
                f"very {phrase}",
                f"extremely {phrase}"
            ])
        
        # Create DataFrame
        mental_health_df = pd.DataFrame({
            'text': mental_health_samples,
            'label': [1] * len(mental_health_samples)
        })
        
        print(f"Created {len(mental_health_df)} mental health shaming samples")
        return mental_health_df
    
    def create_clean_dataset(self):
        """Create comprehensive clean/non-abusive dataset"""
        print("Creating comprehensive clean dataset...")
        
        # Positive words and phrases
        positive_words = [
            'love', 'happy', 'great', 'awesome', 'wonderful', 'amazing', 'excellent',
            'fantastic', 'beautiful', 'kind', 'helpful', 'friend', 'family', 'success',
            'achievement', 'progress', 'hope', 'peace', 'joy', 'smile', 'laugh',
            'good', 'nice', 'sweet', 'caring', 'supportive', 'positive', 'blessed',
            'grateful', 'thankful', 'appreciate', 'cherish', 'treasure', 'brave',
            'strong', 'confident', 'smart', 'intelligent', 'wise', 'clever',
            'healthy', 'well', 'balanced', 'stable', 'calm', 'peaceful'
        ]
        
        # Neutral everyday words
        neutral_words = [
            'hello', 'hi', 'goodbye', 'morning', 'evening', 'night', 'day', 'time',
            'work', 'school', 'home', 'food', 'water', 'sleep', 'rest', 'play',
            'read', 'write', 'talk', 'listen', 'walk', 'run', 'sit', 'stand',
            'computer', 'phone', 'book', 'paper', 'pen', 'table', 'chair', 'door',
            'window', 'car', 'bus', 'train', 'plane', 'road', 'street', 'house'
        ]
        
        # Polite phrases
        polite_phrases = [
            'thank you', 'thanks', 'please', 'excuse me', 'sorry', 'pardon me',
            'good morning', 'good night', 'have a nice day', 'take care', 'be safe',
            'good luck', 'all the best', 'best wishes', 'congratulations', 'well done',
            'great job', 'excellent work', 'amazing effort', 'outstanding performance'
        ]
        
        # Compliments (non-body related)
        compliments = [
            'you are kind', 'you are smart', 'you are talented', 'you are creative',
            'you are brave', 'you are strong', 'you are amazing', 'you are wonderful',
            'you are helpful', 'you are caring', 'you are supportive', 'you are thoughtful',
            'you have a great personality', 'you have a beautiful mind', 'you have a kind heart',
            'you are a good friend', 'you are a great person', 'you are inspiring',
            'you are motivating', 'you are encouraging', 'you are positive',
            'you are mentally strong', 'you have good mental health', 'you are emotionally balanced'
        ]
        
        # Combine all clean categories
        all_clean = positive_words + neutral_words + polite_phrases + compliments
        
        # Create variations for clean words
        clean_samples = []
        
        for clean in all_clean:
            # Single word
            clean_samples.append(clean)
            
            # With punctuation
            clean_samples.extend([
                f"{clean}.",
                f"{clean}!",
                f"{clean}?",
                f"{clean}!!",
                f"{clean}..."
            ])
            
            # In positive sentences
            sentences = [
                f"you are {clean}",
                f"so {clean}",
                f"very {clean}",
                f"really {clean}",
                f"such a {clean}",
                f"what a {clean} day",
                f"have a {clean} time",
                f"{clean} to see you",
                f"feeling {clean}",
                f"stay {clean}"
            ]
            clean_samples.extend(sentences)
        
        # Create DataFrame
        clean_df = pd.DataFrame({
            'text': clean_samples,
            'label': [0] * len(clean_samples)
        })
        
        print(f"Created {len(clean_df)} clean samples")
        return clean_df
    
    def create_contextual_examples(self):
        """Create contextual examples that test model understanding"""
        print("Creating contextual examples...")
        
        # Abusive in different contexts
        contextual_abusive = [
            "you're a fucking mental patient",
            "go to hell you psycho bitch",
            "stupid crazy asshole motherfucker",
            "die you worthless piece of shit",
            "i hope you burn in hell you lunatic",
            "you should kill yourself you schizo",
            "nobody loves you insane whore",
            "fucking retard go die",
            "you're such a psychotic cunt",
            "what a fucking idiot",
            "stupid fucking bitch",
            "damn you to hell mental case",
            "rot in hell crazy asshole",
            "fucking piece of shit",
            "you're dead to me",
            "i will kill you",
            "go hang yourself",
            "stupid fucking faggot",
            "damn nigger die",
            "fucking spic bastard",
            "you're so mental and ugly",
            "what a crazy pig",
            "you look like a mental whale",
            "insane ugly bitch",
            "stupid crazy cow",
            "you belong in a mental institution",
            "go to a psych ward",
            "you need a straightjacket"
        ]
        
        # Clean in similar sentence structures
        contextual_clean = [
            "you're a wonderful person",
            "go to school you genius",
            "smart brilliant student scholar",
            "live you valuable piece of art",
            "i hope you succeed in life",
            "you should love yourself",
            "everybody likes you angel",
            "brilliant genius live well",
            "you're such a star",
            "what a brilliant mind",
            "smart wonderful friend",
            "bless you to heaven",
            "live in heaven angel",
            "brilliant piece of work",
            "you're dear to me",
            "i will help you",
            "go enjoy yourself",
            "smart wonderful friend",
            "bless you live well",
            "brilliant kind person",
            "you are beautiful inside",
            "you have a kind heart",
            "you are talented",
            "you are creative",
            "you are inspiring",
            "you are motivating",
            "you are mentally healthy",
            "you have good mental health",
            "you are emotionally balanced",
            "you are mentally strong"
        ]
        
        # Create DataFrames
        abusive_df = pd.DataFrame({
            'text': contextual_abusive * 2,  # Repeat for balance
            'label': [1] * len(contextual_abusive * 2)
        })
        
        clean_df = pd.DataFrame({
            'text': contextual_clean * 2,  # Repeat for balance
            'label': [0] * len(contextual_clean * 2)
        })
        
        combined_df = pd.concat([abusive_df, clean_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created {len(combined_df)} contextual samples")
        return combined_df
    
    def combine_and_balance_dataset(self, abusive_df, mental_health_df, clean_df, contextual_df):
        """Combine all datasets and ensure perfect balance"""
        print("Combining and balancing datasets...")
        
        all_dfs = [abusive_df, mental_health_df, clean_df, contextual_df]
        
        # Combine all datasets
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Clean text
        combined_df['text'] = combined_df['text'].apply(self.clean_text)
        
        # Remove empty texts
        combined_df = combined_df[combined_df['text'].str.len() > 0]
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        
        # Check balance
        label_counts = combined_df['label'].value_counts()
        print(f"Before balancing: {label_counts.to_dict()}")
        
        # Perfect balancing - ensure equal numbers
        min_count = min(label_counts)
        
        toxic_samples = combined_df[combined_df['label'] == 1].sample(n=min_count, random_state=42)
        safe_samples = combined_df[combined_df['label'] == 0].sample(n=min_count, random_state=42)
        
        combined_df = pd.concat([toxic_samples, safe_samples], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"After balancing: {combined_df['label'].value_counts().to_dict()}")
        print(f"Final dataset: {len(combined_df)} samples")
        
        return combined_df
    
    def create_optimized_features(self, texts):
        """Create optimized TF-IDF features for abusive word detection"""
        if self.vectorizer is None:
            # Optimized vectorizer for abusive word detection
            self.vectorizer = TfidfVectorizer(
                max_features=30000,  # More features for comprehensive coverage
                ngram_range=(1, 5),  # 1-5 grams for patterns
                stop_words='english',
                min_df=1,  # Include all terms
                max_df=0.98,
                sublinear_tf=True,
                analyzer='word',
                norm='l2',  # L2 normalization
                use_idf=True,
                smooth_idf=True
            )
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)
    
    def train_optimized_model(self, df):
        """Train optimized model for comprehensive abusive word detection"""
        print("Training optimized comprehensive abusive word detection model...")
        
        # Split data with stratification
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['label']
        )
        
        # Create features
        X_train = self.create_optimized_features(train_df['text'])
        X_test = self.create_optimized_features(test_df['text'])
        
        y_train = train_df['label']
        y_test = test_df['label']
        
        # Try multiple models to find the best
        models = {
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=3000,
                C=3.0,
                solver='liblinear',
                penalty='l2',
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            )
        }
        
        best_model = None
        best_accuracy = 0
        best_name = ''
        
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
        
        self.model = best_model
        
        print(f"\nBest model: {best_name} with accuracy: {best_accuracy:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Test specific problematic words
        test_words = [
            # Previous problematic words
            'slut', 'whore', 'bitch', 'cunt', 'fuck', 'shit', 'nigger', 'faggot',
            'retard', 'asshole', 'motherfucker', 'dickhead', 'crap', 'bastard',
            # Prostitution and body shaming
            'prostitute', 'hooker', 'escort', 'fat', 'obese', 'ugly', 'disgusting',
            'hideous', 'pig', 'whale', 'cow', 'elephant', 'gross', 'repulsive',
            # NEW: Mental health insults
            'mental', 'psycho', 'psychotic', 'schizo', 'schizophrenic', 'bipolar',
            'insane', 'crazy', 'mad', 'lunatic', 'nutcase', 'nuts', 'bonkers',
            'deranged', 'unstable', 'disturbed', 'sick', 'sicko', 'freak',
            'weirdo', 'psychopath', 'sociopath', 'maniac',
            # Safe words
            'love', 'happy', 'great', 'awesome', 'wonderful', 'excellent', 'beautiful', 'kind'
        ]
        
        print(f"\nTesting specific words:")
        for word in test_words:
            processed = self.clean_text(word)
            features = self.create_optimized_features([processed])
            pred = self.model.predict(features)[0]
            proba = self.model.predict_proba(features)[0]
            confidence = max(proba)
            toxic_prob = proba[1]
            
            # Expected classification
            abusive_words = ['slut', 'whore', 'bitch', 'cunt', 'fuck', 'shit', 'nigger', 'faggot',
                           'retard', 'asshole', 'motherfucker', 'dickhead', 'crap', 'bastard',
                           'prostitute', 'hooker', 'escort', 'fat', 'obese', 'ugly', 'disgusting',
                           'hideous', 'pig', 'whale', 'cow', 'elephant', 'gross', 'repulsive',
                           'mental', 'psycho', 'psychotic', 'schizo', 'schizophrenic', 'bipolar',
                           'insane', 'crazy', 'mad', 'lunatic', 'nutcase', 'nuts', 'bonkers',
                           'deranged', 'unstable', 'disturbed', 'sick', 'sicko', 'freak',
                           'weirdo', 'psychopath', 'sociopath', 'maniac']
            expected = 1 if word in abusive_words else 0
            correct = "✓" if pred == expected else "✗"
            
            print(f"{correct} '{word}' -> Toxic: {bool(pred)}, Confidence: {confidence:.4f}, Expected: {expected}")
        
        return best_accuracy
    
    def save_model(self):
        """Save the trained model"""
        print("Saving comprehensive model...")
        
        joblib.dump(self.model, os.path.join(self.model_dir, 'toxic_classifier.pkl'))
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'vectorizer.pkl'))
        
        print("Model saved successfully!")
    
    def run_training(self):
        """Run the complete training process"""
        print("Starting comprehensive abusive word detection training with mental health insults...")
        
        # Create datasets
        abusive_df = self.create_comprehensive_abusive_dataset()
        mental_health_df = self.create_mental_health_dataset()
        clean_df = self.create_clean_dataset()
        contextual_df = self.create_contextual_examples()
        
        # Combine and balance
        combined_df = self.combine_and_balance_dataset(abusive_df, mental_health_df, clean_df, contextual_df)
        
        # Train optimized model
        accuracy = self.train_optimized_model(combined_df)
        
        # Save model
        self.save_model()
        
        # Save dataset
        combined_df.to_csv(os.path.join(self.data_dir, 'mental_health_dataset.csv'), index=False)
        
        print(f"\nTraining completed!")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Dataset size: {len(combined_df)} samples")
        print(f"Abusive samples: {len(combined_df[combined_df['label'] == 1])}")
        print(f"Clean samples: {len(combined_df[combined_df['label'] == 0])}")
        
        return accuracy

if __name__ == "__main__":
    trainer = MentalHealthAbusiveTrainer()
    accuracy = trainer.run_training()
    print(f"\nFinal comprehensive model accuracy: {accuracy:.4f}")
