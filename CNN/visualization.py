import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import json
from data_loading import load_data
from preprocessing import preprocess_data
from model_evaluation import evaluate_model
from tensorflow.keras.models import load_model

def visualize_data_distribution(articles):
    categories = [article[2] for article in articles]
    unique, counts = np.unique(categories, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def visualize_training_results(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def visualize_model_performance(model, X_test, y_test, label_encoder):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    print(f'Accuracy: {accuracy_score(y_true, y_pred):.4f}')

def main():
    articles = load_data()
    visualize_data_distribution(articles)

    max_num_words = 20000
    max_sequence_length = 1000
    X, y, tokenizer, label_encoder = preprocess_data(articles, max_num_words, max_sequence_length)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = load_model('best_cnn_model.keras')
    visualize_model_performance(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
