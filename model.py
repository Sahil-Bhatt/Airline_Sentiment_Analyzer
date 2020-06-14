import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, model_from_json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
import h5py
import json
from keras.utils import CustomObjectScope
import os

df = pd.read_csv('Tweets.csv', sep = ',')
# print(df.head(10))

tweet_df = df[['text','airline_sentiment']]
tweet_df = tweet_df[tweet_df['airline_sentiment']!='neutral']

# print(tweet_df.head(30))

sentiment_label = tweet_df.airline_sentiment.factorize()


tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
path = os.getcwd()
weight_path = path + '/tweet.h5'
print(weight_path)
# with open('model_in_json.json','r') as f:
#     model_json = json.load(f)

# model = model_from_json(model_json)
# model.load_weights(weight_path)
# json_file = open('model_in_json.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# lmodel = model_from_json(loaded_model_json, custom_objects={'Embedding' : Embedding})
# # load weights into new model
# lmodel.load_weights("tweet.h5")
lmodel = tf.keras.models.load_model(weight_path)


test_sentences = ["This is soo sad", "Very good service", "The flight was late by an hour", "Too many delays, ugh disgusting", "The pilot was well experienced"]

# for text in test_sentences:
#     print(text)
#     tw= tokenizer.texts_to_sequences([text])
#     tw= pad_sequences(tw,maxlen=200)
#     prediction = int(lmodel.predict(tw).round().item())
#     print(sentiment_label[1][prediction])

while True:
    text= input("Enter your sentence: ")
    print(text)
    tw= tokenizer.texts_to_sequences([text])
    tw= pad_sequences(tw,maxlen=200)
    prediction = int(lmodel.predict(tw).round().item())
    print(sentiment_label[1][prediction])