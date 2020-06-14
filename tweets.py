import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
import os
import h5py
import json

df = pd.read_csv('Tweets.csv', sep = ',')
# print(df.head(10))

tweet_df = df[['text','airline_sentiment']]
tweet_df = tweet_df[tweet_df['airline_sentiment']!='neutral']

# print(tweet_df.head(30))

sentiment_label = tweet_df.airline_sentiment.factorize()


tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)

vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(tweet)

padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# print(tokenizer.word_index)

# print(tweet[0])
# print(encoded_docs[0])
# print(padded_sequence[0])

embedding_vector_length = 32
model = Sequential()

model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=5 , batch_size=32)
dir = os.getcwd()
filepath = os.path.join(dir,'tweet.h5')
model_json = model.to_json()
with open("model_in_json.json","w") as json_file:
    json.dump(model_json, json_file)

model.save(filepath)
print("Model weights saved")
