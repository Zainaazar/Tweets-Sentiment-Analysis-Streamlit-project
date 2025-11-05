# Importing Libraries, downloading packages
## Nltk Data Collection
#=====================================
import nltk                                   # Imports the Natural Language Toolkit library for text processing
nltk.download('gutenberg')                    # Downloads the Gutenberg text corpus from NLTK
from nltk.corpus import gutenberg             # Imports the Gutenberg corpus for accessing classic texts like Hamlet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

#Data Manipulation
#=====================================
import pandas as pd                           # Imports pandas for handling and analyzing data in DataFrame format
import numpy as np                            # Imports NumPy for numerical operations and array manipulations

#Keras  & Tensorflow
#=====================================
import tensorflow as tf                                           # Imports TensorFlow (backend for Keras deep learning)
from keras.preprocessing.text import Tokenizer        # Tokenizer: converts text to sequences of integers
from tensorflow.keras.preprocessing.sequence import pad_sequences # pad_sequences: ensures all text sequences have equal length
from keras.models import Sequential 
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau                              # Sequential: used to build models layer-by-layer
from keras.layers import Dense, LSTM, Embedding, Dropout, GRU, Bidirectional, SpatialDropout1D  
                                                     # Imports common neural network layers:
                                                                  # Dense = fully connected layer
                                                                  # LSTM = Long Short-Term Memory (for sequence data)
                                                                  # Embedding = converts word indices into dense vectors
                                                                  # Dropout = prevents overfitting by randomly disabling neurons
                                                                  # GRU = Gated Recurrent Unit (simpler alternative to LSTM)
                                                                  # EarlyStopping: stops training early to prevent overfitting

#Sklearn
#=====================================
from sklearn.model_selection import train_test_split               # Splits data into training and testing sets
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix, ConfusionMatrixDisplay# For evaluating model performance (precision, recall, etc.)

# regular expression
#=======================================
import re

import contractions
import emoji

#gensim
#=======================================
import gensim
from gensim.models import word2vec, KeyedVectors,  Word2Vec
import gensim.downloader as api



import matplotlib.pyplot as plt

import seaborn as sns

#Pickle
#=====================================
import pickle                                                      # Used to save and load Python objects (like models or tokenizers)


nltk.download('punkt')          # For tokenization
nltk.download('stopwords')      # For stopword removal
nltk.download('wordnet')        # For lemmatization (optional)
nltk.download('omw-1.4')        # Support files for wordnet (optional)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

## Loading Data

import kagglehub


# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

print("Path to dataset files:", path)

import os

os.listdir(path)


train_df = pd.read_csv(f"{path}/twitter_training.csv", encoding='latin-1')
test_df = pd.read_csv(f"{path}/twitter_validation.csv", encoding='latin-1')

## Exploring The Data EDA using pandas

train_df.head()

train_df.tail()

train_df.info()

train_df.columns

train_df.drop(['Borderlands', '2401'], axis=1, inplace= True)

train_df.columns = train_df.columns.str.strip()
train_df.rename(columns={"Positive": 'Sentiment', 'im getting on borderlands and i will murder you all ,': 'Tweet' },
                inplace= True)

test_df.head()

test_df.columns

test_df.drop(['3364', 'Facebook'], axis=1, inplace= True)

test_df.columns = test_df.columns.str.strip()
test_df.rename(columns={"Irrelevant": 'Sentiment', 'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tomâs great auntie as âHayley canât get out of bedâ and told to his grandma, who now thinks Iâm a lazy, terrible person ð¤£': 'Tweet' },
                inplace= True)

# Combine for uniform preprocessing
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

combined_df.head()

combined_df.info()

#filling na with most frequent 
most_frequent_tweet = combined_df['Tweet'].mode()[0]
combined_df['Tweet'].fillna(most_frequent_tweet, inplace=True)

combined_df.isnull().sum()

##Preprocessing

# applying on one sample to check the right methods to do 

# picking a random sample
sample= combined_df["Tweet"][300]

sample= sample.lower()

#tokenize
tokenizer= word_tokenize(sample)

tokenizer

#removing Punctuations
punctuations = r'[@#.,!?;:$%^&*()\-_=+{}[\]<>|\\/"]'

after_reg = []
for i in tokenizer:
    reg = re.sub(punctuations, '', i)       # remove punctuation
    reg = re.sub('[^a-zA-Z]', '', reg)      # keep only letters
    after_reg.append(reg)

after_reg


#stemming
after_stem=[]
stemmer= SnowballStemmer("english", ignore_stopwords=True)
for i in after_reg:
  stem= stemmer.stem(i)
  after_stem.append(stem)
print(after_stem)

#Lemmatizing
after_lemm=[]
lemma= WordNetLemmatizer()
for i in after_stem:
  lem= lemma.lemmatize(i)
  after_lemm.append(lem)
print(after_lemm)

# removing stop words
from nltk.corpus import stopwords
stopwords= set(stopwords.words("english"))
after_stopwords=[]
for i in after_lemm:
  if i not in stopwords:
    after_stopwords.append(i)
print(after_stopwords)

# preprocessing the data




from nltk.corpus import stopwords # Import stopwords again

stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}
lemmatizer = WordNetLemmatizer()

corpus = []
for text in combined_df["Tweet"]:
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    corpus.append(" ".join(clean_tokens))

corpus[1:8]

# Word2Vec Model

 # split back into list of words
tokenized_corpus= [sentence.split() for sentence in corpus]

#tarin the word2vec model
model= Word2Vec( sentences= tokenized_corpus, 
        vector_size= 100,
        window= 7,
        min_count= 1,
        workers= 4  
)

vector = model.wv['love']
vector.shape

# Convert texts to sequences of integers
tok= tf.keras.preprocessing.text.Tokenizer()
tok.fit_on_texts(tokenized_corpus)
sequences=tok.texts_to_sequences(tokenized_corpus)
print(sequences[70:84])

vocab_size = len(tok.word_index) + 1  # +1 for padding token
print("Vocabulary size:", vocab_size)

#  Padding Sequences

# for padding need to get the longest sequence that the algorithm found Max Len
max_len_seq= max(len(i) for i in sequences)
max_len_seq

# Padding sequences
padded_sequences= tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen= max_len_seq, padding= "pre" )

type(padded_sequences)

padded_sequences[0]

combined_df['Sentiment'].value_counts()

##Creating XY predictors and labels


x= np.array(padded_sequences)# Map each sentiment string to a unique integer
mapping = {
    'Negative': 0,
    'Positive': 1,
    'Neutral': 2,
    'Irrelevant': 3
}

y = combined_df['Sentiment'].map(mapping)

# Convert to numpy array
y_mapped = np.array(y)

# Convert to categorical with 4 classes
y_cat = tf.keras.utils.to_categorical(y_mapped, num_classes=4)

x_train, x_test, y_train, y_test= train_test_split(x,y_cat, test_size= 0.2, random_state= 42)

# Building the model 

model_hybrid = Sequential()
model_hybrid.add(Embedding(vocab_size, 100, input_length=max_len_seq))
model_hybrid.add(SpatialDropout1D(0.3))
model_hybrid.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))))
model_hybrid.add(GRU(32, kernel_regularizer=l2(0.001)))
model_hybrid.add(Dropout(0.3))
model_hybrid.add(Dense(4, activation="softmax", kernel_regularizer=l2(0.001)))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model_hybrid.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-3)

history = model_hybrid.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,            
    batch_size=32,       
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    validation_split=0.2,
    shuffle=True
)

y_pred= model_hybrid.predict(x_test)

y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
acc_score= accuracy_score(y_test_labels, y_pred_labels)
acc_score

c_r= classification_report(y_test_labels, y_pred_labels)
print(c_r)

model_hybrid.summary()# wont work before model.fit


model_hybrid.save("hybrid_model.h5")

model_hybrid.save("hybrid_model.keras")  # safer and preferred in TF 2.15+


# Retrieve training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
labels = ['Negative', 'Positive', 'Neutral', 'Irrelevant']
# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Convert one-hot encoded to numeric labels
y_test_classes = np.argmax(y_test, axis=1)

# Define the mapping
label_map = {
    0: 'Negative',
    1: 'Positive',
    2: 'Neutral',
    3: 'Irrelevant'
}

# Create DataFrame
df_ytest = pd.DataFrame({'encoded_label': y_test_classes})
df_ytest['class_name'] = df_ytest['encoded_label'].map(label_map)

# Get class counts
class_counts = df_ytest['class_name'].value_counts()
print(class_counts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tok, f)

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))  # correct


from tensorflow.keras.models import load_model
model_hybrid = load_model("hybrid_model.h5")


label_map = {
    0: 'Negative',
    1: 'Positive',
    2: 'Neutral',
    3: 'Irrelevant'
}

def predict_sentiment(text):
    # Clean and tokenize (using the previously fitted tok object)
    seq = tok.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen= max_len_seq, padding='post', truncating='post')

    # Predict (using the trained model_hybrid)
    pred = model_hybrid.predict(padded)
    label_index = np.argmax(pred, axis=1)[0]
    sentiment = label_map[label_index]

    return sentiment

# test the model
print(predict_sentiment("I live In Jordan"))