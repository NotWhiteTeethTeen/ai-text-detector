import pandas as pd
import string
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib

# 1. Load Data
print("Downloading dataset...")
dataset = load_dataset(
    "parquet", 
    data_files="hf://datasets/Hello-SimpleAI/HC3@refs/convert/parquet/all/train/*.parquet", 
    split="train"
)

texts, labels = [], []
for row in dataset.select(range(2000)):
    for ans in row['human_answers']:
        texts.append(ans)
        labels.append(0) 
    for ans in row['chatgpt_answers']:
        texts.append(ans)
        labels.append(1)

df = pd.DataFrame({'text': texts, 'label': labels})

# 2. Extract Features
print("Extracting stylometric features...")
df_feat = df.copy()
df_feat['char_count'] = df_feat['text'].apply(len)
df_feat['word_count'] = df_feat['text'].apply(lambda x: len(x.split()))
df_feat['avg_word_len'] = df_feat['char_count'] / (df_feat['word_count'] + 1)
df_feat['punct_count'] = df_feat['text'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))

X = df_feat[['text', 'char_count', 'word_count', 'avg_word_len', 'punct_count']]
y = df_feat['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build and Train Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text_tfidf', TfidfVectorizer(max_features=5000), 'text'),
        ('num_scaler', StandardScaler(), ['char_count', 'word_count', 'avg_word_len', 'punct_count'])
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("Training model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Evaluation Metrics ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 4))

# 4. Save locally
joblib.dump(pipeline, 'ai_stylometric_pipeline.pkl')
print("\nPipeline saved locally! You can now run your Streamlit app.")