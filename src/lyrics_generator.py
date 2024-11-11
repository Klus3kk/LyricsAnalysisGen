import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_model(total_words, max_sequence_len):
    model = Sequential([
        Embedding(total_words, 64, input_length=max_sequence_len - 1),
        LSTM(100),
        Dense(total_words, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, predictors, labels, epochs=100):
    history = model.fit(predictors, labels, epochs=epochs, verbose=1)
    return history

def generate_lyrics(seed_text, model, tokenizer, max_sequence_len):
    for _ in range(50):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = tokenizer.index_word[np.argmax(predicted)]
        seed_text += " " + output_word
    return seed_text
