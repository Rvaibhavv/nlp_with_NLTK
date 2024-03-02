from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, TFTrainingArguments, TFTrainer
import pandas as pd
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Load and preprocess your dataset
train_df = pd.read_csv('data/train.csv')
train_df = train_df.sample(n=10000, random_state=42)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

train_df['cleaned_text'] = train_df['comment_text'].apply(preprocess_text)

X = list(train_df['cleaned_text'])
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Tokenize the input texts
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encoding = tokenizer(X_train, truncation=True, padding=True)
test_encoding = tokenizer(X_test, truncation=True, padding=True)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encoding), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encoding), y_test))

# Define the training arguments
training_args = TFTrainingArguments(
    output_dir='C:/Users/rvaib/OneDrive/Desktop/nlp_with_NLTK/nlp/content/output',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/content',
    logging_steps=10,
    eval_steps=1
)

# Build the model
with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# Create the trainer and train the model
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)  
trainer.train()
