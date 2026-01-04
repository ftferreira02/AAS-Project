import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os

# Configuration
MAX_LEN = 200  # Max characters to look at in URL
VOCAB_SIZE = 100 # Approx distinct chars in URLs + 1 for OOV

class CharCNN:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def build_model(self):
        self.model = Sequential([
            Input(shape=(MAX_LEN,)),
            # 1. Embedding: Map chars to vectors
            Embedding(input_dim=VOCAB_SIZE + 1, output_dim=32),
            
            # 2. Conv1D: Scan for patterns (like "logjn" or "paypa1")
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            
            # 3. Pooling: Grab the strongest signal from the scan
            GlobalMaxPooling1D(),
            
            # 4. Dense: Classification
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid') # Output 0-1 probability
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, urls, labels, epochs=5, batch_size=64):
        # Create Tokenizer (Character level)
        self.tokenizer = Tokenizer(char_level=True, lower=True, num_words=VOCAB_SIZE)
        self.tokenizer.fit_on_texts(urls)
        
        # Convert URLs to numbers
        sequences = self.tokenizer.texts_to_sequences(urls)
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
        y = np.array(labels)

        print("Training Char-CNN...")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, url):
        if not self.model:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess single URL
        if isinstance(url, str):
            urls = [url]
        else:
            urls = url
            
        seq = self.tokenizer.texts_to_sequences(urls)
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        return self.model.predict(padded).flatten()

    def save(self, path='ml/cnn_data'):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras")) # Use .keras format
        with open(os.path.join(path, "tokenizer.pkl"), 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def load(self, path='ml/cnn_data'):
        self.model = tf.keras.models.load_model(os.path.join(path, "model.keras"))
        with open(os.path.join(path, "tokenizer.pkl"), 'rb') as f:
            self.tokenizer = pickle.load(f)
