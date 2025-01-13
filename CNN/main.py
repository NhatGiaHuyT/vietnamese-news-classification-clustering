from data_loading import load_data
from preprocessing import preprocess_data
from model_training import train_cnn_model
from sklearn.model_selection import train_test_split

# Run model_evaluation.py seperately

def main():
    articles = load_data()
    max_num_words = 20000
    max_sequence_length = 1000
    X, y, tokenizer, label_encoder = preprocess_data(articles, max_num_words, max_sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    train_cnn_model(X_train, y_train, max_num_words, max_sequence_length)

if __name__ == "__main__":
    main()
