from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def preprocess_data(articles, max_num_words=20000, max_sequence_length=1000):
    texts, labels = zip(*[(article[1], article[2]) for article in articles])

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen=max_sequence_length)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.asarray(labels)

    return data, labels, tokenizer, label_encoder