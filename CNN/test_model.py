import json
import numpy as np
import random
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from data_loading import load_data

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
MODEL_PATH = 'best_cnn_model.keras' 
JSON_PATH = 'news_dataset.json'  

def load_model():
    return keras_load_model(MODEL_PATH)

def load_and_preprocess_json(json_path, tokenizer):
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    contents = [item['content'] for item in json_data if 'content' in item]

    if not contents:
        raise ValueError("No content field found in JSON file.")
    
    random_content = random.choice(contents)

    sequences = tokenizer.texts_to_sequences([random_content])
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    return data, random_content

def create_tokenizer(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def create_label_encoder(all_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder

def predict(data, model, label_encoder):
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    return label_encoder.inverse_transform(predicted_labels)

def main():
    articles = load_data() 

    all_texts = [article[1] for article in articles]
    all_labels = [article[2] for article in articles]

    tokenizer = create_tokenizer(all_texts)
    label_encoder = create_label_encoder(all_labels)

    model = load_model()

    data, content = load_and_preprocess_json(JSON_PATH, tokenizer)

    predictions = predict(data, model, label_encoder)

    print(f"Content Snippet: {content[:1000]}...") 
    print(f"Predicted Category: {predictions[0]}\n")

if __name__ == "__main__":
    main()